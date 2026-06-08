
from pathlib import Path

def read_dataset(file_path:Path):
    words = open(file_path).read().splitlines()
    return words