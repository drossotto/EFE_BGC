from pathlib import Path
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.workflow_config.workflow_config import get_logger
from EFE.efe_processing import EFEModule
from scripts.annotation_planes.arg_annotation_plane import build_aligned_arg_labels

logger = get_logger("Annotation Plane Plot")

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

def context_embeddings_to_2d_and_arg_overlay(model_path: Path, bgc_to_idx_path: Path, arg_hits_path: Path, output_path: Path):
    logger.info("Loading model...")
    state_dict = torch.load(model_path, map_location="cpu")
    context_weight = state_dict["context_embeddings.weight"]
    num_bgcs, embedding_dim = context_weight.shape
    domain_weight = state_dict["domain_embeddings.0.weight"]
    num_domains = domain_weight.shape[0]

    model = EFEModule(num_bgcs=num_bgcs, num_domains=num_domains, embedding_dim=embedding_dim)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        context_embeddings = model.context_embeddings.weight.clone()

    autoencoder = Autoencoder2D(input_dim=embedding_dim)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    logger.info("Training autoencoder...")
    for epoch in range(100):
        optimizer.zero_grad()
        output = autoencoder(context_embeddings)
        loss = criterion(output, context_embeddings)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        context_2d = autoencoder.encoder(context_embeddings).numpy()

    with open(bgc_to_idx_path) as f:
        bgc_to_idx = json.load(f)
    binary_labels = build_aligned_arg_labels(arg_hits_path, bgc_to_idx).numpy()

    df = pd.DataFrame({
        "x": context_2d[:, 0],
        "y": context_2d[:, 1],
        "has_ARG": binary_labels
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="x", y="y", hue="has_ARG", palette={0: "gray", 1: "red"}, s=25, alpha=0.7)
    plt.axis("equal")
    plt.title("2D Autoencoded BGC Embeddings with ARG Overlay")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.legend(title="ARG Present")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    logger.info(f"ARG overlay plot saved to {output_path}")
    
if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    context_embeddings_to_2d_and_arg_overlay(
        model_path=base / "results" / "efe_model.pt",
        bgc_to_idx_path=base / "results" /"bgc_to_idx.json",
        arg_hits_path=base / "filtered_rgi_summary.csv",
        output_path=base / "results" / "arg_overlay_plot.png"
    )