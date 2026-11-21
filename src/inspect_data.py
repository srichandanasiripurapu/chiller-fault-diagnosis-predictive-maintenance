
from pathlib import Path

import pandas as pd


# Find the main project folder.
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Locate the CSV file inside the data folder.
DATA_FILE = PROJECT_DIR / "data" / "Chiller_Fault_Diagnosis.csv"


def main():
    print("=" * 60)
    print("CHILLER FAULT DIAGNOSIS - DATA INSPECTION")
    print("=" * 60)

    # Check whether the CSV file exists.
    if not DATA_FILE.exists():
        print("\nERROR: CSV file not found.")
        print(f"Expected location: {DATA_FILE}")
        return

    # Read the CSV file.
    print("\nLoading dataset...")
    df = pd.read_csv(DATA_FILE, sep=";")

    # Display the dataset size.
    print("\n1. DATASET SIZE")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # Display all column names.
    print("\n2. COLUMN NAMES")
    for column in df.columns:
        print(column)

    # Display the first five rows.
    print("\n3. FIRST FIVE ROWS")
    print(df.head().to_string())

    # Display the data types.
    print("\n4. DATA TYPES")
    print(df.dtypes.to_string())

    # Check missing values.
    print("\n5. MISSING VALUES")
    print(df.isnull().sum().to_string())

    # Count duplicate rows.
    print("\n6. DUPLICATE ROWS")
    print(df.duplicated().sum())

    print("\nInspection completed!")


if __name__ == "__main__":
    main()
