from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
BACKGROUNDS_DIR = DATA_DIR / "backgrounds" / "tabletops"
SYNTHETIC_DATASETS_DIR = DATA_DIR / "synthetic_datasets"
REFERENCE_IMAGES_DIR = PROJECT_ROOT / "images"
REFERENCE_ANNOTATIONS_PATH = REFERENCE_IMAGES_DIR / "reference_cards.json"
REAL_TESTSET_DIR = DATA_DIR / "testset"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"
DINOV3_REPO_DIR = PROJECT_ROOT / "src" / "dinov3"
