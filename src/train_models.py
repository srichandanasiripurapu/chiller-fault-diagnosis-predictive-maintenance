
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "processed" / "chiller_sample.parquet"
REPORT_DIR = PROJECT_DIR / "reports"
MODEL_DIR = PROJECT_DIR / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_PER_CLASS = 40_000
TEST_FRACTION = 0.20
RANDOM_STATE = 42

# Only varying sensor measurements are used as features.
FEATURES = [
    "TEI",
    "TCI",
    "TCO",
    "TRE",
    "TRC",
    "TRC_sub",
    "Tsh_suc",
    "TR_dis",
    "TCA",
    "kW",
    "TO_sump",
    "PO_net",
    "PRE",
    "PRC",
    "P_lift",
    "CDW_DT",
    "CHW_DT",
]

TARGET = "fault"

# --------------------------------------------------
# 2. Load a manageable, balanced sample
# --------------------------------------------------
def main():
    print("=" * 60)
    print("CHILLER FAULT CLASSIFICATION")
    print("=" * 60)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Check that data/processed/chiller_sample.parquet exists."
        )

    print("\nReading target, time, and sensor columns...")
    columns = ["time", TARGET] + FEATURES
    df = pd.read_parquet(DATA_FILE, columns=columns)

    print(f"Rows available: {len(df):,}")

    if df[TARGET].isna().any():
        raise ValueError("Target column contains missing labels.")

    if df[FEATURES].isna().any().any():
        raise ValueError("Sensor features contain missing values.")

    # Sample each fault class independently so all classes are represented.
    sampled_parts = []
    for fault_label, group in df.groupby(TARGET, sort=True):
        n = min(SAMPLE_PER_CLASS, len(group))
        sampled_parts.append(
            group.sample(n=n, random_state=RANDOM_STATE)
        )
        print(f"Class {fault_label}: selected {n:,} rows")

    sample = pd.concat(sampled_parts, ignore_index=True)
    del df

    # Create time blocks within each class. All rows in a block stay
    # together, so a block cannot appear in both train and test.
    sample["time_block"] = (
        sample["time"] // 100_000
    ).astype("int32")

    sample["group_id"] = (
        sample[TARGET].astype(str)
        + "_"
        + sample["time_block"].astype(str)
    )

    # Select whole time blocks for the test set, separately for each class.
    rng = np.random.default_rng(RANDOM_STATE)
    test_group_ids = set()

    for fault_label, group in sample.groupby(TARGET):
        groups = group["group_id"].unique()
        if len(groups) < 2:
            raise ValueError(
                f"Not enough distinct time blocks for class {fault_label}."
            )

        shuffled_groups = rng.permutation(groups)
        n_test_groups = max(1, int(round(len(groups) * TEST_FRACTION)))
        test_group_ids.update(shuffled_groups[:n_test_groups])

    is_test = sample["group_id"].isin(test_group_ids)

    train = sample.loc[~is_test]
    test = sample.loc[is_test]

    X_train = train[FEATURES].astype("float32")
    X_test = test[FEATURES].astype("float32")

    # Use a fixed, consistent mapping for all eight fault classes.
    class_names = sorted(
        ["cf", "eo", "fwc", "fwe", "nc", "normal", "rl", "ro"]
    )
    label_to_id = {
        label: index for index, label in enumerate(class_names)
    }
        # Save the mapping used by the trained classifiers.
    import json

    mapping_path = MODEL_DIR / "fault_label_mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "label_to_id": label_to_id,
                "id_to_label": {
                    str(value): key
                    for key, value in label_to_id.items()
                },
            },
            indent=4,
        ),
        encoding="utf-8",
    )
    print(f"Saved label mapping: {mapping_path}")

    y_train = train[TARGET].map(label_to_id).astype("int32")
    y_test = test[TARGET].map(label_to_id).astype("int32")

    print("\nFault label mapping:")
    print(label_to_id)

    del sample, train, test

    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Test rows:     {len(X_test):,}")
    print("\nTraining class counts:")
    print(y_train.value_counts().sort_index())
    print("\nTest class counts:")
    print(y_test.value_counts().sort_index())

    # --------------------------------------------------
    # 3. Define the five classifiers
    # --------------------------------------------------
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=300,
                random_state=RANDOM_STATE,
            ),
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=20,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=2,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=2,
            random_state=RANDOM_STATE,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=150,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.1,
            verbosity=-1,
            n_jobs=2,
            random_state=RANDOM_STATE,
        ),
    }

    # --------------------------------------------------
    # 4. Train, evaluate, and save reports
    # --------------------------------------------------
    results = []

    for name, model in models.items():
        print("\n" + "-" * 60)
        print(f"Training: {name}")
        start = time.perf_counter()

        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test, predictions, average="macro", zero_division=0
        )

        print(f"Accuracy:   {accuracy:.4f}")
        print(f"Macro F1:   {macro_f1:.4f}")
        print(f"Train time: {train_seconds:.1f} seconds")

        results.append({
            "model": name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "train_seconds": train_seconds,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        })

        report = classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
        safe_name = name.lower().replace(" ", "_")
        report_path = REPORT_DIR / f"{safe_name}_report.txt"
        report_path.write_text(
            f"Model: {name}\n"
            f"Accuracy: {accuracy:.6f}\n"
            f"Macro F1: {macro_f1:.6f}\n\n"
            f"{report}\n\nConfusion matrix:\n"
            f"{confusion_matrix(y_test, predictions)}\n",
            encoding="utf-8",
        )

        # Save each fitted model for later analysis.
        model_path = MODEL_DIR / f"{safe_name}.joblib"
        joblib.dump(model, model_path)
        print(f"Saved model: {model_path.name}")
        print(f"Saved report: {report_path.name}")

        del predictions

    # --------------------------------------------------
    # 5. Save and display the comparison
    # --------------------------------------------------
    results_df = pd.DataFrame(results).sort_values(
        "macro_f1", ascending=False
    )
    results_path = REPORT_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON (ranked by macro F1)")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\nComparison saved to: {results_path}")

    best_model_name = results_df.iloc[0]["model"]
    print(f"\nHighest macro F1 in this experiment: {best_model_name}")
    print(
        "\nNote: results apply to this synthetic dataset and this "
        "time-block holdout; they do not establish real-world "
        "industrial performance."
    )


if __name__ == "__main__":
    main()
