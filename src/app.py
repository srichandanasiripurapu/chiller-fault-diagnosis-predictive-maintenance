
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path


# Allow Python to find the project's src folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict_fault import predict_all_models, load_label_mapping
from explain_prediction import explain_prediction

st.set_page_config(
    page_title="Chiller Fault Diagnosis AI",
    page_icon="❄️",
    layout="wide",
)

st.title("❄️ Chiller Fault Diagnosis & Predictive Maintenance AI")
st.write(
    "Enter chiller sensor readings to estimate the current operating condition."
)

st.warning(
    "Research prototype using synthetic data. "
    "Do not use predictions for safety-critical or real-world maintenance decisions."
)

SENSOR_FEATURES = [
    "TEI", "TCI", "TCO", "TRE", "TRC", "TRC_sub",
    "Tsh_suc", "TR_dis", "TCA", "kW", "TO_sump",
    "PO_net", "PRE", "PRC", "P_lift", "CDW_DT", "CHW_DT",
]

st.subheader("Sensor readings")
st.caption("Enter values in the same units and ranges as the training dataset.")

with st.form("sensor_form"):
    readings = {}

    columns = st.columns(3)
    DEFAULT_SENSOR_VALUES = {
    "TEI": 12.0,
    "TCI": 7.0,
    "TCO": 35.0,
    "TRE": 5.0,
    "TRC": 10.0,
    "TRC_sub": 5.0,
    "Tsh_suc": 5.0,
    "TR_dis": 70.0,
    "TCA": 30.0,
    "kW": 100.0,
    "TO_sump": 40.0,
    "PO_net": 300.0,
    "PRE": 500.0,
    "PRC": 1000.0,
    "P_lift": 500.0,
    "CDW_DT": 5.0,
    "CHW_DT": 5.0,
}

    for index, feature in enumerate(SENSOR_FEATURES):
        with columns[index % 3]:
            readings[feature] = st.number_input(
                feature,
                value=float(DEFAULT_SENSOR_VALUES.get(feature, 0.0)),
                format="%.4f",
                key=feature,
            )
    submitted = st.form_submit_button("Diagnose chiller")



if submitted:
    try:
        results = predict_all_models(readings)

        st.subheader("Diagnosis results from all five models")

        for model_name, prediction in results.items():
            st.write(f"**{model_name}**")

            if prediction == "normal":
                st.success("Predicted condition: NORMAL")
            else:
                st.error(f"Predicted fault class: {prediction.upper()}")

    except Exception as error:
        st.error(f"Prediction failed: {error}")


if submitted:
    try:
        explanation, classes = explain_prediction(readings)

        # Find the class predicted by Logistic Regression.
        mapping = load_label_mapping()
        predicted_label = results["Logistic Regression"]
        class_id = mapping["label_to_id"][predicted_label]
        class_index = list(classes).index(class_id)

        # Get the SHAP contribution of each sensor.
        shap_values = explanation.values[0, :, class_index]

        shap_df = pd.DataFrame({
            "Sensor": SENSOR_FEATURES,
            "SHAP value": shap_values,
        })

        # Display the 10 most influential sensors.
        shap_df["Importance"] = shap_df["SHAP value"].abs()
        shap_df = shap_df.nlargest(10, "Importance")
        shap_df = shap_df.sort_values("SHAP value")

        st.subheader("🔍 Why did Logistic Regression make this prediction?")
        st.caption(
            "SHAP values show how each sensor influenced the selected "
            "fault-class score. Positive and negative values indicate "
            "different directions of influence."
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(shap_df["Sensor"], shap_df["SHAP value"])
        ax.set_xlabel("SHAP value")
        ax.set_ylabel("Sensor")
        ax.set_title("Top 10 Sensor Contributions")
        ax.axvline(0, linewidth=0.8)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    except Exception as error:
        st.error(f"SHAP explanation failed: {error}")



st.divider()

st.header("📈 Individual Model Performance")

model_results = {
    "Logistic Regression": {"Accuracy": 99.22, "Macro F1": 99.21},
    "Decision Tree": {"Accuracy": 90.29, "Macro F1": 90.68},
    "Random Forest": {"Accuracy": 96.24, "Macro F1": 96.34},
    "XGBoost": {"Accuracy": 99.13, "Macro F1": 99.13},
    "LightGBM": {"Accuracy": 94.19, "Macro F1": 94.16},
}


for model_name, scores in model_results.items():
    st.subheader(model_name)

    individual_df = pd.DataFrame({
        "Metric": ["Accuracy", "Macro F1"],
        "Score (%)": [
            scores["Accuracy"],
            scores["Macro F1"],
        ],
    })

    st.bar_chart(
        individual_df.set_index("Metric"),
        
        
    )

    st.write(f"Accuracy: {scores['Accuracy']:.2f}%")
    st.write(f"Macro F1: {scores['Macro F1']:.2f}%")


st.caption(
    "These scores are from the initial model comparison experiment "
    "using synthetic data."
)

st.subheader("🏆 Model Comparison")

comparison_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost",
        "LightGBM",
    ],
    "Accuracy (%)": [
        99.22,
        90.29,
        96.24,
        99.13,
        94.19,
    ],
    "Macro F1 (%)": [
        99.21,
        90.68,
        96.34,
        99.13,
        94.16,
    ],
})

st.bar_chart(
    comparison_df.set_index("Model"),
    y=["Accuracy (%)", "Macro F1 (%)"],
    stack=None,
)

st.caption(
    "Results from the initial model comparison experiment. "
    "Evaluation used synthetic data."
)

st.subheader("🔥 Confusion Matrix Heatmap")

fault_labels = ["cf", "eo", "fwc", "fwe", "nc", "normal", "rl", "ro"]

confusion_matrix_data = [
    [7962, 0, 0, 0, 0, 0, 0, 0],
    [0, 7967, 0, 0, 0, 0, 0, 0],
    [0, 0, 7969, 0, 0, 0, 0, 0],
    [0, 0, 0, 8110, 0, 0, 0, 0],
    [0, 0, 0, 0, 7966, 0, 0, 0],
    [0, 0, 1, 3, 41, 7655, 7, 174],
    [0, 0, 0, 0, 0, 0, 7962, 0],
    [0, 0, 0, 0, 0, 0, 0, 7952],
]

fig, ax = plt.subplots(figsize=(10, 7))

sns.heatmap(
    confusion_matrix_data,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=fault_labels,
    yticklabels=fault_labels,
    ax=ax,
)

ax.set_xlabel("Predicted fault")
ax.set_ylabel("Actual fault")
ax.set_title("Logistic Regression — Chronological Validation")

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)


st.header("📊 Model Performance")
st.write(
    "The Logistic Regression model was evaluated using a "
    "chronological holdout split of the synthetic dataset."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Chronological Accuracy", "99.65%")

with col2:
    st.metric("Macro F1 Score", "99.64%")

with col3:
    st.metric("Fault Classes", "8")


st.subheader("Supported fault classes")
st.write("CF, EO, FWC, FWE, NC, NORMAL, RL, RO")

fault_descriptions = {
    "cf": "Fault class CF",
    "eo": "Fault class EO",
    "fwc": "Fault class FWC",
    "fwe": "Fault class FWE",
    "nc": "Fault class NC",
    "normal": "Normal operating class",
    "rl": "Fault class RL",
    "ro": "Fault class RO",
}

fault_table = pd.DataFrame(
    [
        {"Class label": label, "Description": description}
        for label, description in fault_descriptions.items()
    ]
)

st.dataframe(
    fault_table,
    hide_index=True,
    use_container_width=True,
)


fault_descriptions = {
    "cf": "Fault class CF",
    "eo": "Fault class EO",
    "fwc": "Fault class FWC",
    "fwe": "Fault class FWE",
    "nc": "Fault class NC",
    "normal": "Normal operating class",
    "rl": "Fault class RL",
    "ro": "Fault class RO",
}


st.info(
    "These metrics come from chronological validation on synthetic data. "
    "They do not demonstrate equivalent performance on real industrial "
    "equipment. This prototype is not intended for safety-critical decisions."
)

st.caption(
    "Model: Logistic Regression | "
    "Training and evaluation use synthetic chiller data."
)

