
from openai_explainer import generate_fault_explanation

example_readings = {
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

example_shap = {
    "TCO": 0.35,
    "TCA": -0.20,
    "CDW_DT": 0.15,
}

try:
    explanation = generate_fault_explanation(
        predicted_fault="cf",
        sensor_readings=example_readings,
        shap_contributions=example_shap,
    )

    print("\nOPENAI CONNECTION SUCCESSFUL!\n")
    print(explanation)

except Exception as error:
    print("\nOPENAI TEST FAILED:")
    print(error)
