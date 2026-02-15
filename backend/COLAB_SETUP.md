# Colab Setup Instructions

## Quick Start

1. **Upload files to Colab:**
   - `triage_synthetic_dataset_20k.csv`
   - `train_triage_models.py`

2. **Install dependencies:**
```python
!pip install -r requirements.txt
```

Or install individually:
```python
!pip install pandas numpy scikit-learn xgboost lightgbm catboost joblib
```

3. **Run training:**
```python
!python train_triage_models.py
```

## Files Created

After training, the following files will be created in the `models/` directory:
- `preprocessor.joblib` - Data preprocessing pipeline
- `risk_xgb.joblib` - XGBoost model for Risk_Level
- `risk_lgbm.joblib` - LightGBM model for Risk_Level
- `risk_cat.joblib` - CatBoost model for Risk_Level
- `dept_xgb.joblib` - XGBoost model for Department

## Using the Prediction Script

After training, you can use `predict_triage.py`:

```python
from predict_triage import predict_risk_and_department

patient = {
    'Age': 45,
    'Systolic_BP': 120,
    'Heart_Rate': 80,
    'Body_Temperature_C': 37.2,
    'Respiratory_Rate': 18,
    'SpO2': 98,
    'Onset_Duration_Value': 2.5,
    'Onset_Duration_Unit': 'hours',
    'Gender': 'M',
    'Chief_Complaint': 'chest pain'
}

result = predict_risk_and_department(patient)
print(result)
```

## Expected Output

The training script will:
- Load and clean the dataset
- Show data distributions
- Train 3 ensemble models for Risk_Level
- Train 1 model for Department
- Display classification reports
- Save all models
- Test the prediction function

Training time: ~5-10 minutes depending on Colab resources.
