
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

for file_path in sorted(DATA_DIR.glob("*.parquet")):
    print(f"\n{'=' * 60}")
    print(f"File: {file_path.name}")

    df = pd.read_parquet(file_path)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("Column names:")
    print(df.columns.tolist())

    for column in ["fault", "fault_id", "severity_level", "path_label"]:
        if column in df.columns:
            print(f"\n{column} counts:")
            print(df[column].value_counts(dropna=False).head(10).to_string())

    print("\nFirst 2 rows:")
    print(df.head(2).to_string(index=False))
