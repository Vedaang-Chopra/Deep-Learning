from pathlib import Path
def find_project_root(start=None, markers=("pyproject.toml", ".git", "requirements.txt", ".gitignore")):
    p = Path(start or Path.cwd()).resolve()
    for cur in [p, *p.parents]:
        if any((cur / m).exists() for m in markers):
            return cur
    return p

BASE_CODE_DIR_PATH = find_project_root()
DATASET_DIR = BASE_CODE_DIR_PATH / 'datasets'
DATASET_DIR, BASE_CODE_DIR_PATH