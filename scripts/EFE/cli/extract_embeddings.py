import typer
from pathlib import Path
import torch
import pandas as pd
import json
import logging

from scripts.EFE.efe_core import EFEModule

app = typer.Typer(
    help="Extracct embeddings from BGC probabilistic EFE models"
)

logger = logging.getLogger("EFE Extract Embeddings")

@app.command(name="extract-embeddings")
def extract_embeddings_cli(
    model_path: Path = typer.Option(
        ...,
        help="Path to the trained EFE model file"
    ),
    bgc_map_path: Path = typer.Option(
        ...,
        help="Path to the BGC index map JSON"
    ),
    domain_map_path: Path = typer.Option(
        ...,
        help="Path to the domain index map JSON"
    ),
    output_path: Path = typer.Option(
        ...,
        help="Directory to save the extracted embeddings"
    ),
    embedding_dim: int = typer.Option(
        64,
        help="Dimensionality of the embeddings"
    ),
    data_source: str = typer.Option(
        "bgc",
        help="Set to 'bgc' to extract BGC context embeddings, or domain_[value], where value is 0, 85, 170, 255 for specific domain embeddings."
    )
):
    """
    Extracts learned BGC context embeddings or specific domain embeddings from a trained EFE model.
    """

    def load_json_map(
        path: Path,
        map_name: str    
    ):
        try:
            with open(path, "r") as file:
                map_data = json.load(file)
            logger.info(f"Loaded {map_name} map from {path}.")

            return map_data
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {path}")
            raise
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    bgc_to_idx = load_json_map(
        bgc_map_path,
        "BGC"
    )
    idx_to_bgc = {v: k for k, v in bgc_to_idx.items()}
    logger.debug(f"Loaded {len(bgc_to_idx)} BGCs from map.")

    domain_to_idx = load_json_map(
        domain_map_path,
        "Domain"
    )
    idx_to_domain = {v: k for k, v in domain_to_idx.items()}
    logger.debug(f"Loaded {len(domain_to_idx)} domains from map.")

    model = EFEModule(
        len(bgc_to_idx),
        len(domain_to_idx),
        embedding_dim
    ).to(device)

    try:
        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        model.eval()
        logger.info(f"Loaded model from {model_path}.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


    if data_source == "bgc":
        embeddings_tensor = model.context_embeddings.weight.detach().cpu().numpy()
        index = [idx_to_bgc[i] for i in range(len(bgc_to_idx))]
        logger.info("Extracted BGC context embeddings.")

    elif data_source.startswith("domain_"):
        value = data_source.split("_")[1]
        if value not in ["0", "85", "170", "255"]:
            logger.error(f"Invalid domain value: {value}. Must be one of 0, 85, 170, 255.")
            raise ValueError(f"Invalid domain value: {value}")

        embeddings_tensor = model.domain_embeddings[value].weight.detach().cpu().numpy()
        index = [idx_to_domain[i] for i in range(len(domain_to_idx))]
        logger.info(f"Extracted domain embeddings for value {value}.")
    else:
        logger.error(f"Invalid data source: {data_source}. Must be 'bgc' or 'domain_[value]'.")
        raise ValueError(f"Invalid data source: {data_source}")
    
    try:
        embeddings_df = pd.DataFrame(
            embeddings_tensor,
            index=index
        )
        embeddings_df.to_csv(
            output_path,
            sep="\t",
            index=True
        )
        logger.info(f"Saved embeddings to {output_path}.")

    except Exception as e:
        logger.error(f"Failed to save embeddings: {e}")
        raise