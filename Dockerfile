
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY models/ ./models/
COPY reports/ ./reports/

# Streamlit configuration
EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
