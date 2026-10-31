"""Load, validate, and clean the phishing email dataset."""
from pathlib import Path
import pandas as pd

DEFAULT_DATASET = Path(__file__).resolve().parent / "phishing_email.pdf.csv"
TEXT_COLUMN = "text_combined"
TARGET_COLUMN = "label"


def clean_dataset(input_path=DEFAULT_DATASET, output_path=None):
    """Clean a copy of the source data and save it separately."""
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.with_name("cleaned_dataset.csv")
    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    try:
        data = pd.read_csv(input_path)
    except Exception as exc:
        raise ValueError(f"Could not read CSV file {input_path}: {exc}") from exc

    missing_columns = {TEXT_COLUMN, TARGET_COLUMN} - set(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    original_count = len(data)
    data = data[[TEXT_COLUMN, TARGET_COLUMN]].copy()
    data[TEXT_COLUMN] = data[TEXT_COLUMN].astype("string").str.strip()
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    data = data.dropna(subset=[TEXT_COLUMN, TARGET_COLUMN])
    data = data[data[TEXT_COLUMN].str.len() > 0]

    invalid = sorted(set(data[TARGET_COLUMN].unique()) - {0, 1})
    if invalid:
        raise ValueError(f"Expected labels 0 and 1 (0=legitimate, 1=phishing); found {invalid}")

    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype("int8")
    before_deduplication = len(data)
    data = data.drop_duplicates(subset=[TEXT_COLUMN, TARGET_COLUMN]).reset_index(drop=True)
    removed_duplicates = before_deduplication - len(data)

    if data[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Both classes must be present to train and evaluate the model.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Source rows: {original_count:,}; rows removed as missing/blank: {original_count - before_deduplication:,}")
    print(f"Duplicate rows removed: {removed_duplicates:,}; cleaned rows: {len(data):,}")
    print("Class counts (0=legitimate, 1=phishing):")
    print(data[TARGET_COLUMN].value_counts().sort_index().to_string())
    print(f"Cleaned dataset saved to: {output_path}")
    return data


if __name__ == "__main__":
    clean_dataset()
