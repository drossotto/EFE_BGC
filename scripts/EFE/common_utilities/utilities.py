# common_utilities/utilities.py

from pathlib import Path
import random

import torch
from pydantic import BaseModel, field_validator
import pandas as pd

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_cuda_available() -> str:
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

class ArgsValidator(BaseModel):
    input_tsv: Path
    output_dir: Path

    @field_validator("input_tsv", mode="before")
    def validate_input_tsv(cls, input_tsv):
        if not input_tsv.exists() or not input_tsv.is_file():
            raise ValueError(f"Input TSV file {input_tsv} does not exist.")
        try:
            df = pd.read_csv(input_tsv, sep="\t", index_col=0)
        except Exception as e:
            raise ValueError(f"Failed to read TSV file: {e}")
        
        if df.empty:
            raise ValueError("Input TSV file is empty.")
        
        for col, dtype in df.dtypes.items():
            if not pd.api.types.is_numeric_dtype(dtype):
                raise ValueError(f"Column '{col}' is not numeric dtype={dtype}")
        
        valid_values = {0, 85, 170, 255}
        unique_values = set(df.values.flatten())

        if not unique_values.issubset(valid_values):
            raise ValueError(
                f"Input TSV file contains invalid values: {unique_values - valid_values}. "
                f"Only {valid_values} are allowed."
            )
        
        return input_tsv  
    
    @field_validator("output_dir", mode="before")
    def validate_output_dir(cls, output_dir):
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir    
    
def load_bgc_matrix(tsv_file: Path) -> pd.DataFrame:
    """
    Converts the BGC TSV file into a pandas DataFrame.
    Args:
        tsv_file (Path): Path to the TSV file containing BGC data.
    """
    try:
        df = pd.read_csv(tsv_file, sep="\t", index_col=0)
    
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"The file {tsv_file} is empty or not formatted correctly.")
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Error parsing the file {tsv_file}: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while reading the file {tsv_file}: {e}")
    
    return df

def filter_present_domains(
    matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove columns (feature domains) that are zero in all BGCs.
    """
    return matrix.loc[:, (matrix != 0).any(axis=0)]

__all__ = [
    "ArgsValidator", 
    "load_bgc_matrix",
    "filter_present_domains"
]