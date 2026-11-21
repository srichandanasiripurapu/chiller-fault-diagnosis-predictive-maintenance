
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "Chiller_Fault_Diagnosis.csv"
REPORT_FILE = PROJECT_DIR / "reports" / "data_analysis.txt"


def main():
    print("=" * 60)
    print("CHILLER FAULT DIAGNOSIS - DATA ANALYSIS")
    print("=" * 60)

    if not DATA_FILE.exists():
        print(f"ERROR: Dataset not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE, sep=";")

    # 1. Count the fault categories.
    print("\n1. FAULT CATEGORY COUNTS")
    print(df["fault"].value_counts(dropna=False).to_string())

    # 2. Count severity levels.
    print("\n2. SEVERITY LEVEL COUNTS")
    print(df["severity_level"].value_counts(dropna=False).sort_index().to_string())

    # 3. Check the combinations of fault and severity.
    print("\n3. FAULT BY SEVERITY")
    print(
        pd.crosstab(
            df["fault"],
            df["severity_level"],
            margins=True
        ).to_string()
    )

    # 4. Check the source labels.
    print("\n4. SOURCE FILE COUNTS")
    print(df["source_file"].value_counts(dropna=False).to_string())

    # 5. Check the path labels.
    print("\n5. PATH LABEL COUNTS")
    print(df["path_label"].value_counts(dropna=False).to_string())

    # 6. Check how many distinct times are present.
    print("\n6. TIME INFORMATION")
    print(f"Unique time values: {df['time'].nunique()}")
    print(f"Minimum time: {df['time'].min()}")
    print(f"Maximum time: {df['time'].max()}")

    # 7. Save a report.
    report_lines = [
        "CHILLER FAULT DIAGNOSIS - DATA ANALYSIS",
        "=" * 50,
        "\nFAULT CATEGORY COUNTS",
        df["fault"].value_counts(dropna=False).to_string(),
        "\nSEVERITY LEVEL COUNTS",
        df["severity_level"].value_counts(
            dropna=False
        ).sort_index().to_string(),
        "\nFAULT BY SEVERITY",
        pd.crosstab(
            df["fault"],
            df["severity_level"],
            margins=True
        ).to_string(),
        "\nSOURCE FILE COUNTS",
        df["source_file"].value_counts(dropna=False).to_string(),
        "\nPATH LABEL COUNTS",
        df["path_label"].value_counts(dropna=False).to_string(),
        "\nTIME INFORMATION",
        f"Unique times: {df['time'].nunique()}",
        f"Minimum time: {df['time'].min()}",
        f"Maximum time: {df['time'].max()}",
    ]

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nReport saved to: {REPORT_FILE}")
    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()
