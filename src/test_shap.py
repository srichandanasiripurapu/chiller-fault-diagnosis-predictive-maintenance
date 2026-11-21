
from explain_prediction import explain_prediction

example_readings = {
    "TEI": 55.0,
    "TCI": 80.0,
    "TCO": 85.0,
    "TRE": 44.2,
    "TRC": 90.0,
    "TRC_sub": 5.0,
    "Tsh_suc": 28.0,
    "TR_dis": 120.0,
    "TCA": 3.5,
    "kW": 130.0,
    "TO_sump": 130.0,
    "PO_net": 10.0,
    "PRE": 44.0,
    "PRC": 105.0,
    "P_lift": 60.0,
    "CDW_DT": 7.0,
    "CHW_DT": 10.0,
}

explanation, classes = explain_prediction(example_readings)

print("SHAP explanation created successfully!")
print("Available class IDs:", classes)
print("SHAP values shape:", explanation.values.shape)
