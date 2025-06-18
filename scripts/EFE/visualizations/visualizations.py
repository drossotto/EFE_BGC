#!/bin/python3

from pathlib import Path
from typing import List
import json
import random

import torch
import torch.nn as nn
import matplotlib as mpl
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
import typer

from scripts.workflow_config.workflow_config import get_logger
from scripts.EFE.efe_processing import EFEModule

logger = get_logger("EFE Visualization")

def plot_efe_training_history(
    history_tsv: Path,
    output_dir: Path
):
    # Load data
    df = pd.read_csv(history_tsv, sep="\t")
    df["epoch"] = range(1, len(df) + 1)

    # Set publication-quality style
    rcParams["font.family"] = "DejaVu Serif"
    rcParams["axes.labelcolor"] = "black"
    rcParams["xtick.color"] = "black"
    rcParams["ytick.color"] = "black"
    rcParams["axes.edgecolor"] = "black"

    plt.figure(figsize=(6, 4), dpi=450)
    plt.plot(df["epoch"], df["loss"], marker="o", linewidth=1.5, color="black")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Negative Log-Likelihood", fontsize=12)
    plt.title("EFE Training Loss", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "efe_training_loss.png")
    plt.close()

    plt.figure(figsize=(6, 4), dpi=450)
    plt.plot(df["epoch"], df["accuracy"], marker="o", linewidth=1.5, color="black")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("EFE Training Accuracy", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "efe_training_accuracy.png")
    plt.close()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Autoencoder2D(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def plot_kde_contour(df: pd.DataFrame, output_path: Path):
    logger.info("Generating KDE contour overlay...")
    x = df["x"]
    y = df["y"]

    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)(xy)

    df["kde"] = kde

    plt.figure(figsize=(8, 6))
    sns.kdeplot(
        x=df["x"],
        y=df["y"],
        levels=10,
        fill=True,
        cmap="Blues",
        alpha=0.6
    )
    plt.scatter(df["x"], df["y"], s=10, c='black', alpha=0.4)
    plt.title("2D Autoencoded Projection with KDE Contours")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    logger.info(f"KDE overlay saved to {output_path}")

def context_embeddings_to_2d(model_path: Path, output_path: Path, kde_output_path: Path):
    set_seed(84)
    logger.info(f"Loading model from {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")
    context_weight = state_dict["context_embeddings.weight"]
    num_bgcs, embedding_dim = context_weight.shape
    domain_weight = state_dict["domain_embeddings.0.weight"]
    num_domains = domain_weight.shape[0]

    logger.info(f"Inferred: num_bgcs={num_bgcs}, num_domains={num_domains}, embedding_dim={embedding_dim}")

    model = EFEModule(num_bgcs=num_bgcs, num_domains=num_domains, embedding_dim=embedding_dim)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Extracting context embeddings...")
    with torch.no_grad():
        context_embeddings = model.context_embeddings.weight.clone()

    logger.info(f"Context embeddings shape: {context_embeddings.shape}")
    autoencoder = Autoencoder2D(input_dim=embedding_dim)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    logger.info("Training autoencoder for 2D projection...")
    for epoch in range(100):
        optimizer.zero_grad()
        output = autoencoder(context_embeddings)
        loss = criterion(output, context_embeddings)
        loss.backward()
        optimizer.step()

    logger.info("Autoencoder training complete. Projecting to 2D...")
    with torch.no_grad():
        context_2d = autoencoder.encoder(context_embeddings).numpy()

    df = pd.DataFrame(context_2d, columns=["x", "y"])
    plt.figure(figsize=(8, 6))
    sns.kdeplot(
        x=df["x"],
        y=df["y"],
        fill=True,
        cmap="viridis",
        thresh=0.05,
        levels=20
    )
    plt.title("Kernel Density of BGC Context Embeddings (2D Projection)")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    logger.info(f"Saved 2D projection to {output_path}")

    plot_kde_contour(df, kde_output_path)
    logger.info(f"KDE contour overlay saved to {kde_output_path}")


def plot_novelty_scores(
    novelty_scores: pd.DataFrame,
    output_path: Path,
    title: str = "Novelty Scores of BGCs"
):
    logger.info(f"Plotting novelty scores to {output_path}")

    mpl.rcParams['font.family'] = 'DejaVu Serif'

    q1 = novelty_scores["novelty"].quantile(0.25)
    q3 = novelty_scores["novelty"].quantile(0.75)
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr
    top_outliers = (
        novelty_scores[novelty_scores["novelty"] > threshold]
        .nlargest(3, "novelty")
    )
    
    plt.figure(figsize=(10, 4))
    sns.boxplot(
    data=novelty_scores,
    x="novelty",
    color="white",
    fliersize=3,
    linewidth=1.2
    )

    for _, row in top_outliers.iterrows():
        plt.text(
            row["novelty"],
            -0.05,
            row["bgc_id"].split("/")[-1].replace(".gbk", ""),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    plt.title("Novelty Score Distribution", fontsize=12)
    plt.xlabel("Novelty Score")
    plt.yticks([])
    plt.xticks(
        rotation=45, 
        ha='right'
    )
    sns.despine(left=True)
    plt.grid(True, axis='x', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=450
    )

app = typer.Typer(help="Visualization module for EFE modeling output.")

@app.command()
def training_history(
    history_tsv: Path = typer.Option(..., help="Path to training_history.tsv"),
    output_dir: Path = typer.Option(..., help="Directory to save loss and accuracy plots")
):
    plot_efe_training_history(history_tsv, output_dir)

@app.command()
def project_2d(
    model_path: Path = typer.Option(..., help="Path to efe_model.pt"),
    output_path: Path = typer.Option(..., help="Path to save 2D embedding plot"),
    kde_output_path: Path = typer.Option(..., help="Path to save KDE overlay plot")
):
    context_embeddings_to_2d(model_path, output_path, kde_output_path)

@app.command()
def novelty_boxplot(
    novelty_scores_tsv: Path = typer.Option(..., help="Path to efe_novelty_scores.tsv"),
    output_path: Path = typer.Option(..., help="Path to save boxplot")
):
    novelty_scores = pd.read_csv(novelty_scores_tsv, sep="\t")
    plot_novelty_scores(novelty_scores, output_path)

if __name__ == "__main__":
    app()
    
