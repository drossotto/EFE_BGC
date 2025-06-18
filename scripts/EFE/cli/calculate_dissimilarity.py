from pathlib import Path
import logging

import pandas as pd
import typer

from scripts.EFE.efe_core import calculate_dissimilarity

logger = logging.getLogger("Calculate EFE Dissimilarity")

app = typer.Typer(
    help="Calculate dissimilarity scores for BGCs using a trained EFE model."
)

@app.command()
def compute_dissimilar_scores(
    input_tsv: Path = typer.Option(
        ..., 
        help="Path to input BGC-feature TSV"
    ),
    model_path: Path = typer.Option(
        ..., 
        help="Path to trained EFE model (.pt)"
    ),
    output_tsv: Path = typer.Option(
        ..., 
        help="Path to save output TSV with novelty scores"
    )
) -> None:
    logger.info(f"Loading input TSV from {input_tsv}")
    df = pd.read_csv(input_tsv, sep="\t", index_col=0)

    novelty_scores = []
    for idx, row in df.iterrows():
        try:
            score = calculate_dissimilarity(row, model_path)
            logger.debug(f"Calculated novelty for {idx}: {score}")
        except Exception as e:
            logger.warning(f"Error calculating novelty for {idx}: {e}")
            score = float('nan')
        novelty_scores.append((idx, score))

    novelty_df = pd.DataFrame(novelty_scores, columns=["bgc_id", "novelty"])

    logger.info(f"Saving novelty scores to {output_tsv}")
    novelty_df.to_csv(output_tsv, sep="\t", index=False)