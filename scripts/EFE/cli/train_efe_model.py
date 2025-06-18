from pathlib import Path
import json
import logging

import typer
import pandas as pd

from scripts.EFE.efe_core import train_efe_model, save_and_evaluate_model

logger = logging.getLogger("EFE Training")

app = typer.Typer()

@app.command()
def main(
    long_df_path: Path = typer.Option(
        ..., 
        help="Path to the long-form DataFrame TSV"
    ),
    bgc_map_path: Path = typer.Option(
        ..., 
        help="Path to the BGC index map JSON"
    ),
    domain_map_path: Path = typer.Option(
        ..., 
        help="Path to the domain index map JSON"
    ),
    output_dir: Path = typer.Option(
        ..., 
        help="Directory to save the model and training history"
    ),
    embedding_dim: int = 64,
    batch_size: int = 1024,
    epochs: int = 30,
    learning_rate: float = 1e-3
):
    logger.info("Loading long-form DataFrame...")
    df_long = pd.read_csv(long_df_path, sep="\t")

    logger.info("Loading BGC and domain index maps...")
    with open(bgc_map_path) as f:
        bgc_to_idx = json.load(f)
    with open(domain_map_path) as f:
        domain_to_idx = json.load(f)

    logger.info("Beginning training...")
    model, history = train_efe_model(
        df_long=df_long,
        num_bgcs=len(bgc_to_idx),
        num_domains=len(domain_to_idx),
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate
    )

    save_and_evaluate_model(
        model=model,
        history=history,
        output_dir=output_dir
    )