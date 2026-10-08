"""Every path in the project is relative to the paper_data/ folder, found from this file's location.
Renaming or moving the repository does not break anything."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # paper_data/
CONFIGS = ROOT / "configs"
VOCAB = CONFIGS / "vocab.yaml"
DATA = ROOT / "data" / "community"
RECORDS = ROOT / "records"                          # published experiment records (read-only)
OUTPUTS = ROOT / "outputs"                          # anything you run yourself goes here
RESULTS = ROOT / "results"                          # tables and numbers generated from records


def resolve(p) -> Path:
    """A user-given path: absolute stays absolute, relative is taken from paper_data/."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
