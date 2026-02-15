# Triage Prediction API

FastAPI service for medical triage risk assessment and department assignment.

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set environment variable:**
```bash
export GROQ_API_KEY="your-groq-api-key-here"
```

3. **Ensure models are trained:**
```bash
python train_triage_models.py
```

This will create the `models/` directory with all required artifacts.

## Running the API

### Development mode:
```bash
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Production mode:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Health Check
```bash
GET /
GET /health
```

### Standard Prediction
```bash
POST /triage/predict
```

**Request Body:**
```json
{
  "PatientID": "P00001",
  "Age": 45,
  "Gender": "M",
  "Chief_Complaint": "chest pain",
  "Symptoms": "mild discomfort, shortness of breath",
  "Systolic_BP": 120,
  "Heart_Rate": 80,
  "Body_Temperature_C": 37.2,
  "Respiratory_Rate": 18,
  "SpO2": 98,
  "Onset_Duration_Value": 2.5,
  "Onset_Duration_Unit": "hours",
  "Allergies": "None",
  "Current_Medications": "Aspirin",
  "Pre_Existing_Conditions": "Hypertension"
}
```

**Response:**
```json
{
  "risk_level": "Low",
  "risk_source": "ensemble",
  "risk_probabilities": {
    "Low": 0.85,
    "Medium": 0.12,
    "High": 0.03
  },
  "department": "General_Medicine",
  "department_source": "model",
  "rule_reasons": [],
  "top_features": [],
  "nl_explanation": "Based on the patient's stable vital signs and mild symptoms, this case is assessed as low risk. The patient should be seen in General Medicine for routine evaluation of their chest discomfort."
}
```

## Example Usage

### Using curl:
```bash
curl -X POST "http://localhost:8000/triage/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 45,
    "Gender": "M",
    "Chief_Complaint": "chest pain",
    "Symptoms": "mild discomfort",
    "Systolic_BP": 120,
    "Heart_Rate": 80,
    "Body_Temperature_C": 37.2,
    "Respiratory_Rate": 18,
    "SpO2": 98,
    "Onset_Duration_Value": 2.5,
    "Onset_Duration_Unit": "hours"
  }'
```

### Using Python:
```python
import requests

response = requests.post(
    "http://localhost:8000/triage/predict",
    json={
        "Age": 45,
        "Gender": "M",
        "Chief_Complaint": "chest pain",
        "Symptoms": "mild discomfort",
        "Systolic_BP": 120,
        "Heart_Rate": 80,
        "Body_Temperature_C": 37.2,
        "Respiratory_Rate": 18,
        "SpO2": 98,
        "Onset_Duration_Value": 2.5,
        "Onset_Duration_Unit": "hours"
    }
)

print(response.json())
```

## API Documentation

Once the server is running, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Architecture

1. **Rule Engine:** Applies clinical rules first (invalid vitals, high-risk emergencies, chronic stable cases)
2. **ML Ensemble:** If no rules apply, uses XGBoost + LightGBM + CatBoost ensemble for risk prediction
3. **LLM Explanation:** Groq LLaMA-3 generates natural language explanation of the decision

## Error Handling

- If LLM call fails, a fallback explanation is provided
- Missing models return 503 Service Unavailable
- Invalid input returns 400 Bad Request
- All errors are logged for debugging
