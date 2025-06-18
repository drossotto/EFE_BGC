import logging
import pandas as pd
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union
)
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from scripts.EFE.common_utilities.utilities import is_cuda_available, set_seed

logger = logging.getLogger("EFE Processing")

VALUE_TO_CLASS = {0: 0, 85: 1, 170: 2, 255: 3}
CLASS_TO_VALUE = {v: k for k, v in VALUE_TO_CLASS.items()}

def preprocess_matrix_for_efe(
    matrix_df: pd.DataFrame
) -> Tuple[
    pd.DataFrame, 
    Dict[str, int], 
    Dict[str, int]
]:
    logger = logging.getLogger("Preprocessing Matrix")
    logger.info("Preprocessing BGC feature matrix for EFE training...")

    matrix_df = matrix_df.reset_index()
    bgc_id_column = matrix_df.columns[0]  # This will capture the original index name or "index"
    logger.debug(f"Detected BGC ID column: {bgc_id_column}")

    long_df = matrix_df.melt(
        id_vars=[bgc_id_column],
        var_name="domain",
        value_name="value"
    )
    long_df.rename(columns={bgc_id_column: "bgc_id"}, inplace=True)
    long_df["bgc_id"] = long_df["bgc_id"].astype(str)

    logger.debug(f"Long format shape: {long_df.shape}")

    bgc_to_idx = {
        bgc: i
        for i, bgc in enumerate(sorted(long_df["bgc_id"].unique()))
    }
    domain_to_idx = {
        dom: i
        for i, dom in enumerate(sorted(long_df["domain"].unique()))
    }

    logger.info(f"Mapped {len(bgc_to_idx)} BGCs and {len(domain_to_idx)} domains to indices.")

    long_df["bgc_idx"] = long_df["bgc_id"].map(bgc_to_idx)
    long_df["domain_idx"] = long_df["domain"].map(domain_to_idx)

    return long_df, bgc_to_idx, domain_to_idx

class EFEModule(nn.Module):
    """
    Exponential Family Embedding (EFE) module for training on BGC features.
    """
    def __init__(
        self,
        num_bgcs: int,
        num_domains: int,
        embedding_dim: int
    ):
        logger = logging.getLogger("EFEModule Initialization")
        super().__init__()

        logger.info(
            f"Initializing EFEModule with {num_bgcs} BGCs, {num_domains} domains, and "
            f"{embedding_dim}-dimensional embeddings."
        )

        self.context_embeddings = nn.Embedding(num_bgcs, embedding_dim)
        self.domain_embeddings = nn.ModuleDict({
            str(value): nn.Embedding(
                num_embeddings=num_domains,
                embedding_dim=embedding_dim
            ) for value in [0, 85, 170, 255]
        })

    def forward(
        self,
        bgc_idx: torch.Tensor,
        domain_idx: torch.Tensor
    ) -> torch.Tensor:
        if bgc_idx.dtype != torch.long:
            bgc_idx = bgc_idx.long()
        if domain_idx.dtype != torch.long:
            domain_idx = domain_idx.long()

        c_i = self.context_embeddings(bgc_idx)
        logits = []
        for value in [0, 85, 170, 255]:
            eta_j = self.domain_embeddings[str(value)](domain_idx)
            dot = (c_i * eta_j).sum(dim=1, keepdim=True)
            logits.append(dot)
        return torch.cat(logits, dim=1)
    

class EFEDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        logger.info(f"Initializing EFEDataset with {len(df)} samples.")

        self.bgc_idx = torch.tensor(
            df["bgc_idx"].values, dtype=torch.long
        )
        self.domain_idx = torch.tensor(
            df["domain_idx"].values, dtype=torch.long
        )
        self.target = torch.tensor(
            [VALUE_TO_CLASS[v] for v in df["value"].values],
            dtype=torch.long
        )
        logger.debug(
            "Tensorized EFEDataset fields..."
        )

    def __len__(self) -> int:
        return len(self.bgc_idx)
    
    def __getitem__(self, idx: int) -> Tuple[
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor
    ]:
        return (
            self.bgc_idx[idx], 
            self.domain_idx[idx], 
            self.target[idx]
        )

def train_efe_model(
    df_long: pd.DataFrame,
    num_bgcs: int,
    num_domains: int,
    embedding_dim: int = 64,
    batch_size: int = 1024,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    device: Optional[str] = None,
    seed: int = 42
) -> Tuple[EFEModule, list]:
    set_seed(seed)
    # Determine the device (GPU if available, else CPU)
    device = device or is_cuda_available()
    logger.info(f"Training on device: {device}")

    dataset = EFEDataset(df_long)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Consider adjusting based on your CPU cores
        pin_memory=True # Speeds up data transfer to GPU
    )

    model = EFEModule(
        num_bgcs=num_bgcs,
        num_domains=num_domains,
        embedding_dim=embedding_dim
    ).to(device) # Ensure the model is on the selected device

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    best_model_state = None
    best_acc = 0.0

    logger.info(f"Starting training with batch_size={batch_size}, epochs={epochs}, learning_rate={learning_rate}")

    for epoch in range(epochs):
        model.train() # Set model to training mode
        total_loss = 0.0
        correct = 0
        total = 0

        for bgc_idx, domain_idx, target in dataloader:
            # Move data tensors to the specified device
            bgc_idx = bgc_idx.to(device)
            domain_idx = domain_idx.to(device)
            target = target.to(device)

            optimizer.zero_grad() # Zero the gradients
            logits = model(bgc_idx, domain_idx) # Forward pass
            loss = criterion(logits, target) # Calculate loss
            loss.backward() # Backward pass
            optimizer.step() # Update weights

            total_loss += loss.item() * bgc_idx.size(0)
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        history.append((avg_loss, accuracy))

        logger.info(
            "Epoch {}/{} — Loss: {:.4f}, Accuracy: {:.4f}".format(
                epoch + 1, epochs, avg_loss, accuracy
            )
        )

        if accuracy > best_acc:
            best_acc = accuracy
            best_model_state = model.state_dict()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded model state from best epoch with accuracy: {best_acc:.4f}")

    return model, history

def save_and_evaluate_model(
    model: EFEModule,
    history: List[Tuple[float, float]],
    output_dir: Path = Path(__file__).parent,
    model_filename: str = "efe_model.pt"
) -> None:
    """
    Save trained model and training history, and log best epoch stats.
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / model_filename
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to: {model_path.resolve()}")

    if not history:
        logger.warning("Empty training history. Skipping evaluation.")
        return

    history_df = pd.DataFrame(history, columns=["loss", "accuracy"])
    history_path = output_dir / "training_history.tsv"
    history_df.to_csv(history_path, sep="\t", index=False)
    logger.info(f"Training history saved to: {history_path.resolve()}")

    if history_df["accuracy"].isnull().all():
        logger.warning("No valid accuracy values found — skipping best epoch reporting.")
        return

    best_idx = history_df["accuracy"].idxmax()
    best_epoch = int(best_idx) + 1
    best_acc = history_df.loc[best_idx, "accuracy"]
    best_loss = history_df.loc[best_idx, "loss"]

    logger.info(
        f"Best Epoch: {best_epoch} — Accuracy: {best_acc:.4f} — Loss: {best_loss:.4f}"
    )

def calculate_dissimilarity(
    x: Union[pd.Series, torch.Tensor],
    model_path: Path
) -> float:
    state_dict = torch.load(model_path, map_location="cpu")
    num_bgcs, embedding_dim = state_dict["context_embeddings.weight"].shape
    num_domains = state_dict["domain_embeddings.0.weight"].shape[0]

    if isinstance(x, pd.Series):
        x = pd.to_numeric(x, errors="coerce").fillna(0).astype(int)
        x = torch.tensor(x.values, dtype=torch.int64)
    elif isinstance(x, torch.Tensor):
        x = x.clone().detach().to(dtype=torch.int64)
    else:
        raise TypeError("Input must be a pandas Series or a torch Tensor.")

    # Pad or trim
    if x.shape[0] < num_domains:
        x = torch.cat([x, torch.zeros(num_domains - x.shape[0], dtype=torch.int64)])
    elif x.shape[0] > num_domains:
        x = x[:num_domains]

    y = torch.tensor([VALUE_TO_CLASS.get(int(val), -1) for val in x], dtype=torch.int64)
    if (y == -1).any():
        raise ValueError("Input vector contains values not in {0, 85, 170, 255}.")

    domain_indices = torch.arange(num_domains, dtype=torch.int64)

    model = EFEModule(num_bgcs=num_bgcs, num_domains=num_domains, embedding_dim=embedding_dim)
    model.load_state_dict(state_dict)
    model.eval()

    c_star = torch.randn(embedding_dim, requires_grad=True)
    optimizer = torch.optim.Adam([c_star], lr=0.05)

    for _ in range(100):
        optimizer.zero_grad()
        logits = model(torch.zeros(num_domains, dtype=torch.long), domain_indices)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(torch.zeros(num_domains, dtype=torch.long), domain_indices)
        nll = torch.nn.functional.cross_entropy(logits, y, reduction="sum")

    return nll.item()
