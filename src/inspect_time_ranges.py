
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

for file_path in sorted(DATA_DIR.glob("*.parquet")):
    df = pd.read_parquet(
        file_path,
        columns=["time", "fault", "severity_level"]
    )

    print(f"\n{'=' * 45}")
    print(f"File: {file_path.name}")

    print("Overall time range:")
    print(f"  Minimum: {df['time'].min()}")
    print(f"  Maximum: {df['time'].max()}")

    print("\nTime range by severity:")
    summary = df.groupby("severity_level")["time"].agg(
        ["min", "max", "count"]
    )
    print(summary.to_string())

    del df

print("\nTime-range inspection finished.")
