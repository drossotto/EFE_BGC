#!/bin/python3

from pathlib import Path
import pandas as pd
import typer
import json
from pydantic import ValidationError
import traceback

from scripts.common_utilities.utilities import ArgsValidator, load_bgc_matrix
from scripts.workflow_config.workflow_config import get_logger
from scripts.EFE.efe_core import preprocess_matrix_for_efe

logger = get_logger("EFE Load Matrix")
app = typer.Typer(help="Load BGC matrix for EFE model input.")

@app.command()
def main(
    input_tsv: Path = typer.Option(
        ..., 
        help="Path to the input BGC-feature matrix"
    ),
    output_dir: Path = typer.Option(
        ..., 
        help="Directory to save output files"
    )
):
    try:
        args = ArgsValidator(input_tsv=input_tsv, output_dir=output_dir)
    except ValidationError as e:
        logger.error("Argument validation error:\n" + str(e))
        logger.debug(traceback.format_exc())
        raise typer.Exit(code=1)

    logger.info(f"Loading input matrix from {args.input_tsv}")
    df = load_bgc_matrix(args.input_tsv)

    logger.info("Preprocessing matrix...")
    long_df, bgc_to_idx, domain_to_idx = preprocess_matrix_for_efe(df)

    output_dir.mkdir(parents=True, exist_ok=True)

    long_df_path = output_dir / "efe_long_df.tsv"
    bgc_map_path = output_dir / "bgc_to_idx.json"
    domain_map_path = output_dir / "domain_to_idx.json"

    logger.info(f"Saving long-form DataFrame to {long_df_path}")
    long_df.to_csv(long_df_path, sep="\t", index=False)

    logger.info(f"Saving BGC index mapping to {bgc_map_path}")
    with open(bgc_map_path, "w") as f:
        json.dump({str(k): int(v) for k, v in bgc_to_idx.items()}, f)

    logger.info(f"Saving domain index mapping to {domain_map_path}")
    with open(domain_map_path, "w") as f:
        json.dump({str(k): int(v) for k, v in domain_to_idx.items()}, f)

    logger.info("Done.")

if __name__ == "__main__":
    app()