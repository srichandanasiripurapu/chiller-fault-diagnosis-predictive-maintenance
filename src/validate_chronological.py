
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "processed" / "chiller_sample.parquet"
MAPPING_FILE = PROJECT_DIR / "models" / "fault_label_mapping.json"
REPORT_DIR = PROJECT_DIR / "reports"

SAMPLE_PER_CLASS = 40_000
RANDOM_STATE = 42
TRAIN_FRACTION = 0.80

FEATURES = [
    "TEI", "TCI", "TCO", "TRE", "TRC", "TRC_sub",
    "Tsh_suc", "TR_dis", "TCA", "kW", "TO_sump",
    "PO_net", "PRE", "PRC", "P_lift", "CDW_DT", "CHW_DT",
]

def main():
    print("Loading dataset...")
    df = pd.read_parquet(
        DATA_FILE, columns=["time", "fault"] + FEATURES
    )

    print("Creating a balanced sample...")
    parts = []
    for label, group in df.groupby("fault", sort=True):
        n = min(SAMPLE_PER_CLASS, len(group))
        parts.append(group.sample(n=n, random_state=RANDOM_STATE))

    sample = pd.concat(parts, ignore_index=True)
    del df

    with MAPPING_FILE.open("r", encoding="utf-8-sig") as file:
        mapping = json.load(file)

    # Sort time blocks within each fault class.
    sample["time_block"] = (sample["time"] // 100_000).astype("int32")
    train_indices = []
    test_indices = []

    for label, group in sample.groupby("fault"):
        blocks = np.sort(group["time_block"].unique())

        if len(blocks) < 2:
            raise ValueError(f"Not enough time blocks for class {label}")

        split_at = max(1, int(len(blocks) * TRAIN_FRACTION))
        split_at = min(split_at, len(blocks) - 1)

        earlier_blocks = set(blocks[:split_at])
        later_blocks = set(blocks[split_at:])

        train_indices.extend(
            group.index[group["time_block"].isin(earlier_blocks)]
        )
        test_indices.extend(
            group.index[group["time_block"].isin(later_blocks)]
        )

    train = sample.loc[train_indices]
    test = sample.loc[test_indices]

    X_train = train[FEATURES].astype("float32")
    X_test = test[FEATURES].astype("float32")
    y_train = train["fault"].map(mapping["label_to_id"]).astype("int32")
    y_test = test["fault"].map(mapping["label_to_id"]).astype("int32")

    print(f"Training rows: {len(train):,}")
    print(f"Testing rows:  {len(test):,}")
    print("Training a fresh Logistic Regression model...")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=300, random_state=RANDOM_STATE),
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    ids = list(range(len(mapping["id_to_label"])))
    names = [mapping["id_to_label"][str(i)] for i in ids]

    report = classification_report(
        y_test,
        predictions,
        labels=ids,
        target_names=names,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=ids)

    print(f"\nChronological accuracy: {accuracy:.4f}")
    print(f"Chronological macro F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(report)
    print("\nConfusion matrix (actual rows, predicted columns):")
    print(pd.DataFrame(matrix, index=names, columns=names))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "chronological_validation.txt"
    output.write_text(
        f"Accuracy: {accuracy:.6f}\n"
        f"Macro F1: {macro_f1:.6f}\n\n"
        f"{report}\n\nConfusion matrix (actual rows, predicted columns):\n"
        f"{pd.DataFrame(matrix, index=names, columns=names)}\n",
        encoding="utf-8",
    )
    print(f"\nReport saved to: {output}")
    print("This experiment does not overwrite the saved model.")
    print("Results apply to the synthetic dataset, not real-world equipment.")

if __name__ == "__main__":
    main()
