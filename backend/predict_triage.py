"""
Prediction script for triage models.
Loads saved models and provides prediction function.
"""

import pandas as pd
import numpy as np
import joblib
import os


# Model paths
MODEL_DIR = "models"
PREPROCESSOR_PATH = f"{MODEL_DIR}/preprocessor.joblib"
RISK_XGB_PATH = f"{MODEL_DIR}/risk_xgb.joblib"
RISK_LGBM_PATH = f"{MODEL_DIR}/risk_lgbm.joblib"
RISK_CAT_PATH = f"{MODEL_DIR}/risk_cat.joblib"
DEPT_XGB_PATH = f"{MODEL_DIR}/dept_xgb.joblib"
RISK_LABEL_ENCODER_PATH = f"{MODEL_DIR}/risk_label_encoder.joblib"
DEPT_LABEL_ENCODER_PATH = f"{MODEL_DIR}/dept_label_encoder.joblib"


def load_models():
    """Load all saved models and preprocessor."""
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model directory '{MODEL_DIR}' not found. Please train models first.")
    
    print("Loading models...")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    risk_xgb = joblib.load(RISK_XGB_PATH)
    risk_lgbm = joblib.load(RISK_LGBM_PATH)
    risk_cat = joblib.load(RISK_CAT_PATH)
    dept_xgb = joblib.load(DEPT_XGB_PATH)
    risk_label_encoder = joblib.load(RISK_LABEL_ENCODER_PATH)
    dept_label_encoder = joblib.load(DEPT_LABEL_ENCODER_PATH)
    
    risk_pipelines = {
        'xgb': risk_xgb,
        'lgbm': risk_lgbm,
        'cat': risk_cat
    }
    
    print("✓ All models loaded successfully")
    return preprocessor, risk_pipelines, dept_xgb, risk_label_encoder, dept_label_encoder


def predict_risk_and_department(patient_dict):
    """
    Predict risk level and department for a single patient.
    
    Args:
        patient_dict: Dictionary with patient features. Must include:
            - Age (int)
            - Systolic_BP (int)
            - Heart_Rate (int)
            - Body_Temperature_C (float)
            - Respiratory_Rate (int)
            - SpO2 (int)
            - Onset_Duration_Value (float)
            - Onset_Duration_Unit (str: 'hours', 'days', 'months', 'years')
            - Gender (str: 'M', 'F', 'T')
            - Chief_Complaint (str)
    
    Returns:
        dict with:
            - 'risk_level' (str): Predicted risk level
            - 'risk_probabilities' (dict): Probability for each risk class
            - 'department' (str): Predicted department
    """
    # Load models (lazy loading)
    if not hasattr(predict_risk_and_department, '_models_loaded'):
        preprocessor, risk_pipelines, dept_pipeline, risk_label_encoder, dept_label_encoder = load_models()
        predict_risk_and_department.preprocessor = preprocessor
        predict_risk_and_department.risk_pipelines = risk_pipelines
        predict_risk_and_department.dept_pipeline = dept_pipeline
        predict_risk_and_department.risk_label_encoder = risk_label_encoder
        predict_risk_and_department.dept_label_encoder = dept_label_encoder
        predict_risk_and_department._models_loaded = True
    
    preprocessor = predict_risk_and_department.preprocessor
    risk_pipelines = predict_risk_and_department.risk_pipelines
    dept_pipeline = predict_risk_and_department.dept_pipeline
    risk_label_encoder = predict_risk_and_department.risk_label_encoder
    dept_label_encoder = predict_risk_and_department.dept_label_encoder
    
    # Convert to DataFrame
    patient_df = pd.DataFrame([patient_dict])
    
    # Required features
    required_features = [
        'Age', 'Systolic_BP', 'Heart_Rate', 'Body_Temperature_C',
        'Respiratory_Rate', 'SpO2', 'Onset_Duration_Value',
        'Onset_Duration_Unit', 'Gender', 'Chief_Complaint'
    ]
    
    # Check for missing features
    missing = [f for f in required_features if f not in patient_df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    
    # Get ensemble risk probabilities
    prob_xgb = risk_pipelines['xgb'].predict_proba(patient_df)[0]
    prob_lgbm = risk_pipelines['lgbm'].predict_proba(patient_df)[0]
    prob_cat = risk_pipelines['cat'].predict_proba(patient_df)[0]
    
    # Average probabilities
    prob_ensemble = (prob_xgb + prob_lgbm + prob_cat) / 3.0
    
    # Decode predictions using label encoder
    original_classes = risk_label_encoder.classes_
    risk_level_encoded = np.argmax(prob_ensemble)
    risk_level = risk_label_encoder.inverse_transform([risk_level_encoded])[0]
    risk_probabilities = {cls: float(prob) for cls, prob in zip(original_classes, prob_ensemble)}
    
    # Predict department
    dept_encoded = dept_pipeline.predict(patient_df)[0]
    department = dept_label_encoder.inverse_transform([dept_encoded])[0]
    
    return {
        'risk_level': risk_level,
        'risk_probabilities': risk_probabilities,
        'department': department
    }


# Example usage
if __name__ == "__main__":
    # Example patient
    example_patient = {
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
    
    try:
        result = predict_risk_and_department(example_patient)
        print("Prediction Result:")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Risk Probabilities: {result['risk_probabilities']}")
        print(f"  Department: {result['department']}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train_triage_models.py first to train and save the models.")
