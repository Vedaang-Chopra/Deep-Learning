from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE


def find_project_root(start=None, markers=(".gitignore",)):
    p = Path(start or Path.cwd()).resolve()
    for cur in [p, *p.parents]:
        if any((cur / m).exists() for m in markers):
            return cur
    return p


BASE_CODE_DIR_PATH = find_project_root()
DATASET_DIR = BASE_CODE_DIR_PATH / 'datasets'


# Hugging Face resolves this to its user-level default cache unless overridden
# by a standard Hugging Face environment variable.
HF_CACHE_DIR = Path(HF_HUB_CACHE)
