"""Load Iris data, clean it, and create reproducible train/test CSV files."""
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"


def ensure_dataset(path: Path = DATA_PATH) -> None:
    """Create data/data.csv from scikit-learn's Iris dataset if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        iris = load_iris(as_frame=True)
        df = iris.frame.rename(columns={"target": "target"})
        df.to_csv(path, index=False)


def preprocess(raw_path: str | Path = DATA_PATH,
               output_dir: str | Path = PROCESSED_DIR) -> None:
    raw_path = Path(raw_path)
    output_dir = Path(output_dir)
    if not raw_path.exists():
        ensure_dataset(raw_path)

    df = pd.read_csv(raw_path).dropna().drop_duplicates()
    if "target" not in df.columns:
        raise ValueError("Expected a 'target' column in data.csv")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["target"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    print(f"Preprocessed {len(df)} rows; train={len(train_df)}, test={len(test_df)}")


if __name__ == "__main__":
    ensure_dataset()
    preprocess()
