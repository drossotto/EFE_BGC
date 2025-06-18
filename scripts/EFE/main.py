import typer

from scripts.EFE.cli import extract_embeddings
from scripts.EFE.cli import load_for_efe
from scripts.EFE.cli import train_efe_model
from scripts.EFE.cli import calculate_dissimilarity

app = typer.Typer(
    help="Perform Exponential Feature Embedding on Biosynthetic Gene Clusters (BGCs)",
)

app.add_typer(
    extract_embeddings.app,
    name="extract-embeddings"
)

app.add_typer(
    load_for_efe.app,
    name="load-for-efe"
)

app.add_typer(
    train_efe_model.app,
    name="train-efe-model"
)

app.add_typer(
    calculate_dissimilarity.app,
    name="calculate-dissimilarity"
)


if __name__ == "__main__":
    app()