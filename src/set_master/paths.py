from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DATASETS_DIR = DATA_DIR / "synthetic_datasets"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"
DINOV3_REPO_DIR = PROJECT_ROOT / "src" / "dinov3"
