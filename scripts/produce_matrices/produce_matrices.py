#!/bin/python3
from pathlib import Path
from typing import Tuple

import typer
from pydantic import ValidationError
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd
import numpy as np

from scripts.workflow_config.workflow_config import get_logger
from scripts.EFE.cli.load_for_efe import ArgsValidator
from scripts.common_utilities.utilities import load_bgc_matrix, filter_present_domains

logger = get_logger("Heatmap Plot")

def global_sparsity(
    matrix_df: pd.DataFrame
) -> float:
    """
    Calculate the global sparsity of the matrix.
    """
    total_elements = matrix_df.size
    zero_elements = (matrix_df == 0).sum().sum()
    return zero_elements / total_elements

def row_sparsity(
    matrix_df: pd.DataFrame
) -> Tuple[pd.Series, float]:
    """
    Calculate the sparsity of each row, representing a BGC in the matrix.
    """
    series = (matrix_df == 0).sum(axis=1) / matrix_df.shape[1]
    mean_sparsity = series.mean()

    return series, mean_sparsity

def column_sparsity(
    matrix_df: pd.DataFrame
) -> Tuple[pd.Series, float]:
    """
    Calculate the sparsity of each column, representing a feature domain in the matrix.
    """
    series = (matrix_df == 0).sum(axis=0) / matrix_df.shape[0]
    mean_sparsity = series.mean()

    return series, mean_sparsity

def write_sparsity_summary(matrix: pd.DataFrame, output_dir: Path) -> None:
    # Sparsity summary
    global_s = global_sparsity(matrix)
    _, row_mean = row_sparsity(matrix)
    _, col_mean = column_sparsity(matrix)

    summary_df = pd.DataFrame({
        "Metric": ["Global Sparsity", "Mean Row Sparsity", "Mean Column Sparsity"],
        "Value": [global_s, row_mean, col_mean]
    })
    summary_file = output_dir / "sparsity_summary.tsv"
    summary_df.to_csv(summary_file, sep="\t", index=False)

    # Value breakdown
    values, counts = np.unique(matrix.values, return_counts=True)
    total = matrix.size
    breakdown_df = pd.DataFrame({
        "Value": values.astype(int),
        "Count": counts,
        "Percent": [round(c / total, 4) for c in counts]
    })
    breakdown_file = output_dir / "sparsity_value_breakdown.tsv"
    breakdown_df.to_csv(breakdown_file, sep="\t", index=False)   

def plot_bgc_heatmap(
    matrix_df: pd.DataFrame,
    output_file: Path,
    figsize: tuple = (12, 2)
) -> None:
    
    colors = ["white", "#cccccc", "#555555", "black"]
    colormap = ListedColormap(colors)
    bounds = [-1, 1, 86, 171, 256]
    norm = BoundaryNorm(boundaries=bounds, ncolors=len(colors), clip=True)

    fig, ax = plt.subplots(figsize=figsize)

    cax = ax.imshow(matrix_df.values, aspect='auto', cmap=colormap, norm=norm)

    ax.set_xlabel("Feature Domains", fontname="Times New Roman")
    ax.set_ylabel("Biosynthetic Gene Clusters", fontname="Times New Roman")
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(cax, ax=ax, shrink=0.6, pad=0.02, ticks=[0.5, 43, 128, 213])
    cbar.ax.set_yticklabels(["Absent", "Low-Ranked Match", "Sub-Pfam Hit", "Top-Ranked Core Domain Match"])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

heatmap_app = typer.Typer(
    help="Produce matrices for heatmap visualization from BGC feature vectors."
)

@heatmap_app.command("heatmap-plot")
def heatmap_plot(
    input_tsv: Path = typer.Option(
        "--input-tsv",
        help="Input TSV file containing BGC feature vectors.",
    ),
    output_dir: Path = typer.Option(
        "--output-dir",
        help="Directory to save the output heatmap matrices.",
    ),
) -> None:
    """
    Generate heatmap matrices from BGC feature vectors
    """
    try:
        args = ArgsValidator(
            input_tsv=input_tsv,
            output_dir=output_dir
        )
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise typer.Exit(code=1)
    
    logger.info(f"Input TSV: {args.input_tsv}")
    logger.info(f"Output Directory: {args.output_dir}")

    df = load_bgc_matrix(args.input_tsv)
    logger.info(f"Loaded BGC matrix with shape {df.shape}")
    
    df = filter_present_domains(df)
    logger.info(f"Filtered BGC matrix with shape {df.shape}")

    output_file = args.output_dir / "bgc_heatmap.png"
    logger.info(f"Saving heatmap to {output_file}")

    plot_bgc_heatmap(
        matrix_df=df,
        output_file=output_file,
        figsize=(12, 2)
    )

    logger.info("Heatmap plot completed successfully.")

    sparsity_file = args.output_dir / "sparsity_summary.tsv"
    logger.info(f"Writing sparsity summary to {sparsity_file}")
    

    values, counts = np.unique(df.values, return_counts=True)
    for v, c in zip(values, counts):
        logger.info(f"Value: {v}, Count: {c}, Percent: {c / df.size:.4f}")

    logger.info("Writing sparsity summaries...")
    write_sparsity_summary(df, args.output_dir)
    logger.info("Sparsity summaries written successfully.")

    logger.info("Sparsity summary written successfully.")

if __name__ == "__main__":
    logger.info("Starting heatmap plotting")
    heatmap_app()
    logger.info("Heatmap plotting completed")