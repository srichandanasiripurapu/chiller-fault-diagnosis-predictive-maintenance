
import os

from dotenv import load_dotenv
from openai import OpenAI

# Load the API key from the project's .env file.
load_dotenv()

def generate_fault_explanation(
    predicted_fault,
    sensor_readings,
    shap_contributions,
):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OpenAI API key not found. "
            "Please check your .env file."
        )

    client = OpenAI(api_key=api_key)

    sensor_summary = "\n".join(
        f"- {sensor}: {value:.4f}"
        for sensor, value in sensor_readings.items()
    )

    shap_summary = "\n".join(
        f"- {sensor}: {value:+.4f}"
        for sensor, value in shap_contributions.items()
    )

    prompt = f"""
    Explain this chiller fault prediction in simple language.

    Predicted fault class: {predicted_fault}

    Sensor readings:
    {sensor_summary}

    SHAP contributions:
    {shap_summary}

    Please explain:
    1. What the predicted fault means.
    2. Which sensors influenced the prediction most.
    3. What the SHAP contributions indicate.
    4. What a qualified technician could investigate.

    Important:
    - Do not invent sensor thresholds or measurements.
    - Do not claim that the fault is confirmed.
    - Explain that this is a machine-learning prediction.
    - Do not provide safety-critical repair instructions.
    """

    response = client.responses.create(
        model="gpt-5.5",
        input=prompt,
        max_output_tokens=500,
    )

    return response.output_text
