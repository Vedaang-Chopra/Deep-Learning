from pathlib import Path

# Shared datasets live at the Deep-Learning repo root: <root>/datasets
REPO_ROOT = Path(__file__).resolve().parents[4]
DATASETS_ROOT = REPO_ROOT / "datasets"
BASE_CODE_DIR_PATH = DATASETS_ROOT
DATASET_DIR = DATASETS_ROOT