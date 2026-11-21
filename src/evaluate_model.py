
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "processed" / "chiller_sample.parquet"
MODEL_FILE = PROJECT_DIR / "models" / "logistic_regression.joblib"
MAPPING_FILE = PROJECT_DIR / "models" / "fault_label_mapping.json"
REPORT_DIR = PROJECT_DIR / "reports"

SAMPLE_PER_CLASS = 40_000
TEST_FRACTION = 0.20
RANDOM_STATE = 42

FEATURES = [
    "TEI", "TCI", "TCO", "TRE", "TRC", "TRC_sub",
    "Tsh_suc", "TR_dis", "TCA", "kW", "TO_sump",
    "PO_net", "PRE", "PRC", "P_lift", "CDW_DT", "CHW_DT",
]

def main():
    print("Loading dataset...")
    df = pd.read_parquet(DATA_FILE, columns=["time", "fault"] + FEATURES)

    print("Recreating the original balanced sample...")
    parts = []
    for fault_label, group in df.groupby("fault", sort=True):
        parts.append(
            group.sample(n=min(SAMPLE_PER_CLASS, len(group)),
                         random_state=RANDOM_STATE)
        )
    sample = pd.concat(parts, ignore_index=True)
    del df

    print("Recreating the original time-block split...")
    sample["time_block"] = (sample["time"] // 100_000).astype("int32")
    sample["group_id"] = (
        sample["fault"].astype(str) + "_" + sample["time_block"].astype(str)
    )

    rng = np.random.default_rng(RANDOM_STATE)
    test_group_ids = set()

    for fault_label, group in sample.groupby("fault"):
        groups = group["group_id"].unique()
        shuffled_groups = rng.permutation(groups)
        n_test_groups = max(1, int(round(len(groups) * TEST_FRACTION)))
        test_group_ids.update(shuffled_groups[:n_test_groups])

    test = sample.loc[sample["group_id"].isin(test_group_ids)].copy()
    del sample

    with MAPPING_FILE.open("r", encoding="utf-8-sig") as file:
        mapping = json.load(file)

    X_test = test[FEATURES].astype("float32")
    y_true = test["fault"].astype(str)

    y_true_ids = y_true.map(mapping["label_to_id"])
    if y_true_ids.isna().any():
        raise ValueError("Found a fault label missing from the mapping.")

    print("Loading saved model...")
    model = joblib.load(MODEL_FILE)
    y_pred_ids = model.predict(X_test).astype(int)

    accuracy = accuracy_score(y_true_ids, y_pred_ids)
    macro_f1 = f1_score(
        y_true_ids, y_pred_ids, average="macro", zero_division=0
    )

    class_ids = list(range(len(mapping["id_to_label"])))
    class_names = [mapping["id_to_label"][str(i)] for i in class_ids]

    print(f"\nTest rows: {len(test):,}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_true_ids,
        y_pred_ids,
        labels=class_ids,
        target_names=class_names,
        zero_division=0,
    ))

    cm = confusion_matrix(y_true_ids, y_pred_ids, labels=class_ids)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_{name}" for name in class_names],
        columns=[f"Predicted_{name}" for name in class_names],
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "logistic_regression_re_evaluation.txt"
    report_path.write_text(
        f"Accuracy: {accuracy:.6f}\nMacro F1: {macro_f1:.6f}\n\n"
        + classification_report(
            y_true_ids,
            y_pred_ids,
            labels=class_ids,
            target_names=class_names,
            zero_division=0,
        )
        + "\nConfusion matrix:\n"
        + cm_df.to_string(),
        encoding="utf-8",
    )

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(cm_df.to_string())
    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    main()
