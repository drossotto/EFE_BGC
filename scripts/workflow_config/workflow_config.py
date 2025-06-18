import logging
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger named the name assigned.
    """
    if name is None:
        name = __name__
    return logging.getLogger(name)

__all__ = [
    'get_logger',
]