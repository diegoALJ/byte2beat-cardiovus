"""Central configuration for the CardioVUS project."""

from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

REPORTS_DIR = ROOT_DIR / "reports"

# Reproducibility
SEED = 42

# Biological target
GENE_SYMBOL = "KCNH2"
DISEASE_NAME = "Long QT syndrome type 2"

# Create generated directories when the module is imported
for directory in [
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    EMBEDDINGS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
