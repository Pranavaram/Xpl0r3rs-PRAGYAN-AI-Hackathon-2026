"""
Triage Inference Module
Combines rule-based triage, ML ensemble predictions, and LLM explanation generation.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, List, Tuple, Optional
import os
import shap
from operations import get_current_load, choose_operationally_aware_department
from comorbidity import (
    parse_comorbidities,
    evaluate_comorbidities,
    get_comorbidity_risk_adjustment,
    get_comorbidity_preferred_department
)


# ============================================================================
# Configuration
# ============================================================================

MODEL_DIR = "models"
PREPROCESSOR_PATH = f"{MODEL_DIR}/preprocessor.joblib"
RISK_XGB_PATH = f"{MODEL_DIR}/risk_xgb.joblib"
RISK_LGBM_PATH = f"{MODEL_DIR}/risk_lgbm.joblib"
RISK_CAT_PATH = f"{MODEL_DIR}/risk_cat.joblib"
DEPT_XGB_PATH = f"{MODEL_DIR}/dept_xgb.joblib"
RISK_LABEL_ENCODER_PATH = f"{MODEL_DIR}/risk_label_encoder.joblib"
DEPT_LABEL_ENCODER_PATH = f"{MODEL_DIR}/dept_label_encoder.joblib"

# Feature columns used in training
ALL_FEATURES = [
    'Age', 'Systolic_BP', 'Heart_Rate', 'Body_Temperature_C',
    'Respiratory_Rate', 'SpO2', 'Onset_Duration_Value',
    'Onset_Duration_Unit', 'Gender', 'Chief_Complaint'
]


# ============================================================================
# A. Rule-based Engine
# ============================================================================

def apply_rules(patient: dict) -> dict:
    """
    Apply rule-based triage logic to patient data.
    
    Args:
        patient: dict with raw fields (Age, Gender, Chief_Complaint, Symptoms,
                 Systolic_BP, Heart_Rate, Body_Temperature_C, Respiratory_Rate, SpO2,
                 Onset_Duration_Value, Onset_Duration_Unit, Allergies, Current_Medications,
                 Pre_Existing_Conditions)
    
    Returns:
        dict with:
            - forced: bool (whether rules override ML)
            - risk_level: Optional[str] (Low/Medium/High or None)
            - department: Optional[str] (department name or None)
            - rule_reasons: List[str] (list of reason strings)
    """
    # Initialize return structure
    result = {
        "forced": False,
        "risk_level": None,
        "department": None,
        "rule_reasons": []
    }
    
    # Helper to safely get numeric values
    def get_float(key: str, default: float = 0.0) -> float:
        val = patient.get(key)
        if val is None or val == '':
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
    
    def get_str(key: str, default: str = '') -> str:
        val = patient.get(key)
        if val is None:
            return default
        return str(val).lower()
    
    # Extract values
    heart_rate = get_float('Heart_Rate')
    respiratory_rate = get_float('Respiratory_Rate')
    systolic_bp = get_float('Systolic_BP')
    spo2 = get_float('SpO2')
    body_temp = get_float('Body_Temperature_C')
    onset_value = get_float('Onset_Duration_Value')
    onset_unit = get_str('Onset_Duration_Unit')
    chief_complaint = get_str('Chief_Complaint')
    pre_existing_conditions = patient.get('Pre_Existing_Conditions', '')
    
    # Parse comorbidities
    comorbidities = parse_comorbidities(str(pre_existing_conditions))
    
    # Evaluate comorbidity interactions
    comorbidity_effects = evaluate_comorbidities(comorbidities, chief_complaint)
    comorbidity_risk_adjustment = get_comorbidity_risk_adjustment(comorbidities, chief_complaint)
    comorbidity_preferred_dept = get_comorbidity_preferred_department(comorbidities, chief_complaint)
    
    # Rule 1: Invalid vitals (non-positive values)
    if heart_rate <= 0 or respiratory_rate <= 0 or systolic_bp <= 0:
        result["forced"] = True
        result["risk_level"] = "Medium"
        result["department"] = "General_Medicine"
        result["rule_reasons"] = ["Invalid vital signs (non-positive values). Ask to recheck inputs."]
        return result
    
    # Rule 2: Hard high-risk emergency (override ML)
    reasons = []
    
    # Check critical vitals
    if systolic_bp < 80:
        reasons.append("Systolic BP < 80 (shock risk)")
    
    if spo2 < 88:
        reasons.append("SpO2 < 88 (severe hypoxia)")
    
    if heart_rate > 140 or heart_rate < 40:
        reasons.append("Extreme heart rate")
    
    if respiratory_rate > 35 or respiratory_rate < 8:
        reasons.append("Extreme respiratory rate")
    
    # Check for severe emergency keywords in chief complaint
    severe_keywords = [
        "severe chest pain",
        "crushing chest pain",
        "acute chest pain",
        "acute shortness of breath",
        "difficulty breathing",
        "stroke",
        "sudden weakness",
        "loss of consciousness"
    ]
    
    has_severe_keyword = any(keyword in chief_complaint for keyword in severe_keywords)
    if has_severe_keyword:
        reasons.append("Severe emergency keyword in chief complaint")
    
    # Add comorbidity risk adjustment to high-risk determination
    # If comorbidity adds significant risk (>=3 points), consider it in emergency decision
    if comorbidity_risk_adjustment >= 3:
        reasons.append(f"High-risk comorbidity interaction (adds {comorbidity_risk_adjustment} risk points)")
    
    # If any high-risk reasons found, force High risk + Emergency
    # But prefer comorbidity department if specified and risk is high
    if reasons:
        result["forced"] = True
        result["risk_level"] = "High"
        # Use comorbidity preferred department if available and risk is high, otherwise Emergency
        if comorbidity_preferred_dept and comorbidity_risk_adjustment >= 2:
            result["department"] = comorbidity_preferred_dept
            reasons.append(f"Comorbidity preference: {comorbidity_preferred_dept}")
        else:
            result["department"] = "Emergency"
        result["rule_reasons"] = reasons
        return result
    
    # Rule 3: Chronic but stable
    # Check if onset is chronic (months or years)
    is_chronic = onset_unit in ["months", "years"]
    
    # Check if vitals are stable
    stable_bp = 90 <= systolic_bp <= 150
    stable_hr = 60 <= heart_rate <= 110
    stable_temp = 36 <= body_temp <= 38
    stable_spo2 = 92 <= spo2 <= 100
    
    # Check that chief complaint doesn't have severe keywords
    no_severe_keywords = not has_severe_keyword
    
    if is_chronic and stable_bp and stable_hr and stable_temp and stable_spo2 and no_severe_keywords:
        result["forced"] = True
        result["risk_level"] = "Low"
        result["department"] = "General_Medicine"
        result["rule_reasons"] = ["Chronic symptoms with stable vitals"]
        return result
    
    # No rules apply
    return result


# ============================================================================
# B. Loading Models & Preprocessor
# ============================================================================

def load_artifacts(model_dir: str = "models") -> Dict[str, Any]:
    """
    Loads preprocessor, risk models, department model, and label encoders.
    
    Args:
        model_dir: Directory containing saved models
    
    Returns:
        dict with keys:
            - preprocessor
            - risk_xgb, risk_lgbm, risk_cat
            - dept_xgb
            - risk_label_encoder, dept_label_encoder
    """
    artifacts = {}
    
    # Update paths if custom model_dir provided
    if model_dir != "models":
        preprocessor_path = f"{model_dir}/preprocessor.joblib"
        risk_xgb_path = f"{model_dir}/risk_xgb.joblib"
        risk_lgbm_path = f"{model_dir}/risk_lgbm.joblib"
        risk_cat_path = f"{model_dir}/risk_cat.joblib"
        dept_xgb_path = f"{model_dir}/dept_xgb.joblib"
        risk_label_encoder_path = f"{model_dir}/risk_label_encoder.joblib"
        dept_label_encoder_path = f"{model_dir}/dept_label_encoder.joblib"
    else:
        preprocessor_path = PREPROCESSOR_PATH
        risk_xgb_path = RISK_XGB_PATH
        risk_lgbm_path = RISK_LGBM_PATH
        risk_cat_path = RISK_CAT_PATH
        dept_xgb_path = DEPT_XGB_PATH
        risk_label_encoder_path = RISK_LABEL_ENCODER_PATH
        dept_label_encoder_path = DEPT_LABEL_ENCODER_PATH
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' not found. Please train models first.")
    
    # Load preprocessor
    artifacts["preprocessor"] = joblib.load(preprocessor_path)
    
    # Load risk models (extract classifier from pipeline)
    risk_xgb_pipe = joblib.load(risk_xgb_path)
    risk_lgbm_pipe = joblib.load(risk_lgbm_path)
    risk_cat_pipe = joblib.load(risk_cat_path)
    
    # Extract classifiers from pipelines
    artifacts["risk_xgb"] = risk_xgb_pipe.named_steps['classifier']
    artifacts["risk_lgbm"] = risk_lgbm_pipe.named_steps['classifier']
    artifacts["risk_cat"] = risk_cat_pipe.named_steps['classifier']
    
    # Store pipelines for preprocessing
    artifacts["risk_xgb_pipe"] = risk_xgb_pipe
    artifacts["risk_lgbm_pipe"] = risk_lgbm_pipe
    artifacts["risk_cat_pipe"] = risk_cat_pipe
    
    # Load department model
    dept_xgb_pipe = joblib.load(dept_xgb_path)
    artifacts["dept_xgb"] = dept_xgb_pipe.named_steps['classifier']
    artifacts["dept_xgb_pipe"] = dept_xgb_pipe
    
    # Load label encoders
    artifacts["risk_label_encoder"] = joblib.load(risk_label_encoder_path)
    artifacts["dept_label_encoder"] = joblib.load(dept_label_encoder_path)
    
    # Initialize SHAP explainer for XGBoost risk model (once at startup)
    try:
        artifacts["shap_explainer"] = shap.TreeExplainer(artifacts["risk_xgb"])
    except Exception as e:
        # If SHAP initialization fails, continue without it
        artifacts["shap_explainer"] = None
    
    return artifacts


# ============================================================================
# C. SHAP Explainability
# ============================================================================

def get_top_shap_features(
    patient_df: pd.DataFrame,
    preprocessor,
    xgb_model,
    shap_explainer,
    risk_label_encoder,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Get top contributing features using SHAP values for a single patient.
    
    Args:
        patient_df: one-row DataFrame with raw feature columns
        preprocessor: fitted ColumnTransformer
        xgb_model: XGBoost model (not pipeline)
        shap_explainer: SHAP TreeExplainer
        risk_label_encoder: LabelEncoder for risk levels
        top_k: number of top features to return
    
    Returns:
        List of (feature_name, shap_value) tuples sorted by |shap_value| descending.
        Feature names are human-readable (e.g., 'Systolic_BP', 'Chief Complaint: chest pain').
    """
    if shap_explainer is None:
        return []
    
    try:
        # Transform patient data using preprocessor
        X_transformed = preprocessor.transform(patient_df)
        
        # Get SHAP values for the transformed input
        shap_values = shap_explainer.shap_values(X_transformed)
        
        # Handle multi-class output (SHAP returns array for each class)
        if isinstance(shap_values, list):
            # For multi-class, use mean absolute SHAP values across classes for feature importance
            shap_values = np.array(shap_values)
            shap_values = np.mean(np.abs(shap_values), axis=0)
        else:
            # Single output, use as is
            # Check if it has class dimension (e.g. n_samples, n_features, n_classes)
            if hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
                 # Average over class dimension (axis 2)
                 shap_values = np.mean(np.abs(shap_values), axis=2)
            
            shap_values = shap_values[0] if (hasattr(shap_values, 'shape') and len(shap_values.shape) > 1 and shap_values.shape[0] == 1) else shap_values
        
        # Ensure shap_values is 1D
        if hasattr(shap_values, 'shape') and len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        # Get feature names from preprocessor
        feature_names = []
        
        if hasattr(preprocessor, "get_feature_names_out"):
            try:
                feature_names = list(preprocessor.get_feature_names_out())
            except Exception:
                pass
        
        if not feature_names:
            try:
                # 1. Numeric (First transformer)
                numeric_transformer = preprocessor.transformers_[0]
                if numeric_transformer[0] == 'num':
                    feature_names.extend(numeric_transformer[2])
                
                # 2. Categorical (Second transformer)
                cat_transformer = preprocessor.transformers_[1]
                if cat_transformer[0] == 'cat':
                    cat_cols = cat_transformer[2]
                    cat_encoder = preprocessor.named_transformers_['cat']
                    if hasattr(cat_encoder, 'get_feature_names_out'):
                        feature_names.extend(cat_encoder.get_feature_names_out(cat_cols))
                    else:
                        for i, col in enumerate(cat_cols):
                            feature_names.append(f"{col}_encoded_{i}")

                # 3. Text TF-IDF (Indices 2 and 3 in new script)
                # Check for txt_cc and txt_sym transformers
                for name, trans, _ in preprocessor.transformers_:
                    if name == 'txt_cc' and hasattr(trans, 'get_feature_names_out'):
                         feature_names.extend([f"CC_{f}" for f in trans.get_feature_names_out()])
                    elif name == 'txt_sym' and hasattr(trans, 'get_feature_names_out'):
                         feature_names.extend([f"Sym_{f}" for f in trans.get_feature_names_out()])

            except Exception as e:
                print(f"Warning: Feature name extraction failed: {e}")

        # Ensure we have the right number of feature names
        if len(feature_names) != len(shap_values):
            # If mismatch, try to pad or truncate (common in dev), or just use generic
            if len(feature_names) < len(shap_values):
                 feature_names.extend([f"feature_{i}" for i in range(len(feature_names), len(shap_values))])
            else:
                 feature_names = [f"feature_{i}" for i in range(len(shap_values))]
        
        # Create list of (feature_name, shap_value) tuples
        feature_contributions = list(zip(feature_names, shap_values))
        
        # Sort by absolute SHAP value (descending)
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Return top K features
        top_features = feature_contributions[:top_k]
        
        # Make feature names more human-readable
        readable_features = []
        for name, value in top_features:
            # Remove common prefixes from ColumnTransformer
            clean_name = name
            for prefix in ['num__', 'cat__', 'remainder__']:
                if clean_name.startswith(prefix):
                    clean_name = clean_name.replace(prefix, "", 1)
            
            # Clean up OneHotEncoder feature names
            if '_' in clean_name and not clean_name.startswith("Onset_Duration"): 
                # Check if it's a categorical feature with encoded value
                # (Skip Onset_Duration_Value/Unit as they have underscores naturally)
                found_cat = False
                for cat_col in ['Gender', 'Chief_Complaint']:
                    if clean_name.startswith(f"{cat_col}_"):
                        # Format: "Chief_Complaint_chest pain" -> "Chief Complaint: chest pain"
                        value_part = clean_name.replace(f"{cat_col}_", "", 1)
                        readable_name = f"{cat_col.replace('_', ' ')}: {value_part}"
                        found_cat = True
                        break
                
                if not found_cat:
                     readable_name = clean_name.replace('_', ' ')
            else:
                readable_name = clean_name.replace('_', ' ')
            
            readable_features.append((readable_name, float(value)))
            
        return readable_features
    
    except Exception as e:
        # If SHAP computation fails, return empty list (silently fail for production)
        # Error details can be logged if needed
        return []


# ============================================================================
# D. Ensemble Prediction
# ============================================================================

def predict_with_ensemble(patient: dict, artifacts: Dict[str, Any]) -> dict:
    """
    Predict risk level and department using ML ensemble models.
    
    Args:
        patient: raw feature dict (same keys as training features)
        artifacts: output from load_artifacts
    
    Returns:
        dict with:
            - risk_level: str
            - risk_probabilities: Dict[str, float]
            - department: str
            - top_features: List[Tuple[str, float]] (placeholder for SHAP)
    """
    # Build one-row DataFrame with feature columns
    patient_df = pd.DataFrame([patient])
    
    # Ensure all required features are present
    for feat in ALL_FEATURES:
        if feat not in patient_df.columns:
            raise ValueError(f"Missing required feature: {feat}")
    
    # Use pipelines for preprocessing and prediction
    risk_xgb_pipe = artifacts["risk_xgb_pipe"]
    risk_lgbm_pipe = artifacts["risk_lgbm_pipe"]
    risk_cat_pipe = artifacts["risk_cat_pipe"]
    dept_xgb_pipe = artifacts["dept_xgb_pipe"]
    risk_label_encoder = artifacts["risk_label_encoder"]
    dept_label_encoder = artifacts["dept_label_encoder"]
    
    # Get ensemble risk probabilities
    prob_xgb = risk_xgb_pipe.predict_proba(patient_df)[0]
    prob_lgbm = risk_lgbm_pipe.predict_proba(patient_df)[0]
    prob_cat = risk_cat_pipe.predict_proba(patient_df)[0]
    
    # Average probabilities
    prob_ensemble = (prob_xgb + prob_lgbm + prob_cat) / 3.0
    
    # Decode predictions using label encoder
    original_classes = risk_label_encoder.classes_
    risk_level_encoded = np.argmax(prob_ensemble)
    risk_level = risk_label_encoder.inverse_transform([risk_level_encoded])[0]
    risk_probabilities = {cls: float(prob) for cls, prob in zip(original_classes, prob_ensemble)}
    
    # Predict department probabilities (not just class)
    dept_proba = dept_xgb_pipe.predict_proba(patient_df)[0]
    dept_classes = dept_label_encoder.classes_
    dept_clinical_probs = {cls: float(prob) for cls, prob in zip(dept_classes, dept_proba)}
    
    # Get top SHAP features for explainability
    top_features: List[Tuple[str, float]] = []
    if "shap_explainer" in artifacts and artifacts["shap_explainer"] is not None:
        top_features = get_top_shap_features(
            patient_df=patient_df,
            preprocessor=artifacts["preprocessor"],
            xgb_model=artifacts["risk_xgb"],
            shap_explainer=artifacts["shap_explainer"],
            risk_label_encoder=risk_label_encoder,
            top_k=5
        )
    
    return {
        "risk_level": risk_level,
        "risk_probabilities": risk_probabilities,
        "department_clinical_probs": dept_clinical_probs,
        "top_features": top_features
    }


# ============================================================================
# E. LLM Explanation Stub
# ============================================================================

def build_llm_explanation_prompt(patient: dict,
                                 risk_level: str,
                                 risk_probabilities: Optional[Dict[str, float]],
                                 department: str,
                                 rule_reasons: List[str],
                                 top_features: List[Tuple[str, float]],
                                 dept_clinical_probs: Optional[Dict[str, float]] = None,
                                 dept_operational_scores: Optional[Dict[str, float]] = None,
                                 dept_load: Optional[Dict[str, float]] = None,
                                 comorbidity_effects: Optional[List[Dict[str, Any]]] = None,
                                 deterioration_info: Optional[Dict[str, Any]] = None) -> str:
    """
    Build a prompt string to send to an LLM for explanation generation.
    
    Args:
        patient: patient data dict
        risk_level: final risk level (Low/Medium/High)
        risk_probabilities: dict of class -> probability (None if rule-based)
        department: assigned department
        rule_reasons: list of rule-based reasons (empty if ML-based)
        top_features: list of (feature_name, impact) tuples (empty for now)
    
    Returns:
        prompt string for LLM
    """
    # Extract key patient information
    age = patient.get('Age', 'Unknown')
    gender = patient.get('Gender', 'Unknown')
    chief_complaint = patient.get('Chief_Complaint', 'Not specified')
    symptoms = patient.get('Symptoms', 'Not specified')
    systolic_bp = patient.get('Systolic_BP', 'Unknown')
    heart_rate = patient.get('Heart_Rate', 'Unknown')
    body_temp = patient.get('Body_Temperature_C', 'Unknown')
    respiratory_rate = patient.get('Respiratory_Rate', 'Unknown')
    spo2 = patient.get('SpO2', 'Unknown')
    
    # Build prompt
    prompt = f"""You are a medical triage assistant. Explain the triage decision for the following patient in 2-3 simple sentences that a triage nurse can understand. Do not mention probabilities or technical details.

Patient Summary:
- Age: {age}, Gender: {gender}
- Chief Complaint: {chief_complaint}
- Symptoms: {symptoms}
- Vital Signs: BP {systolic_bp}, HR {heart_rate}, Temp {body_temp}°C, RR {respiratory_rate}, SpO2 {spo2}%

Triage Decision:
- Risk Level: {risk_level}
- Assigned Department: {department}
"""
    
    # Add rule-based reasons if present
    if rule_reasons:
        prompt += f"\nRule-Based Factors:\n"
        for reason in rule_reasons:
            prompt += f"- {reason}\n"
    
    # Add ML probabilities if available
    if risk_probabilities:
        prompt += f"\nRisk Assessment:\n"
        for level, prob in risk_probabilities.items():
            prompt += f"- {level}: {prob:.1%}\n"
    
    # Add top SHAP features if available
    if top_features:
        prompt += f"\nTop Model Factors:\n"
        for feature, shap_value in top_features:
            # Classify SHAP value impact
            abs_value = abs(shap_value)
            if abs_value > 0.3:
                impact_desc = "strongly increases" if shap_value > 0 else "strongly decreases"
            elif abs_value > 0.15:
                impact_desc = "moderately increases" if shap_value > 0 else "moderately decreases"
            elif abs_value > 0.05:
                impact_desc = "slightly increases" if shap_value > 0 else "slightly decreases"
            else:
                impact_desc = "minimally affects"
            
            # Get actual patient value for the feature if possible
            feature_lower = feature.lower()
            patient_value = None
            if 'systolic_bp' in feature_lower or 'bp' in feature_lower:
                patient_value = patient.get('Systolic_BP')
            elif 'heart_rate' in feature_lower or 'hr' in feature_lower:
                patient_value = patient.get('Heart_Rate')
            elif 'spo2' in feature_lower or 'oxygen' in feature_lower:
                patient_value = patient.get('SpO2')
            elif 'temperature' in feature_lower or 'temp' in feature_lower:
                patient_value = patient.get('Body_Temperature_C')
            elif 'respiratory' in feature_lower or 'rr' in feature_lower:
                patient_value = patient.get('Respiratory_Rate')
            elif 'age' in feature_lower:
                patient_value = patient.get('Age')
            
            if patient_value is not None:
                prompt += f"- {feature} = {patient_value} ({impact_desc} risk)\n"
            else:
                prompt += f"- {feature} ({impact_desc} risk)\n"
    
    # Add operational load information if available
    if dept_load and dept_clinical_probs and dept_operational_scores:
        prompt += f"\nOperational Context:\n"
        prompt += "Current department loads:\n"
        for dept, load in sorted(dept_load.items(), key=lambda x: x[1], reverse=True):
            from operations import get_load_description
            load_desc = get_load_description(load)
            prompt += f"- {dept}: {load:.1%} ({load_desc})\n"
        
        # Find top clinical departments
        top_clinical = sorted(dept_clinical_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        prompt += f"\nClinical recommendations (top 3):\n"
        for dept, prob in top_clinical:
            prompt += f"- {dept}: {prob:.1%}\n"
        
        prompt += f"\nNote: The system balances clinical safety (primary) with operational efficiency. "
        prompt += f"For high-risk patients, clinical recommendation always takes priority. "
        prompt += f"For lower-risk cases, the system may recommend a department with similar clinical probability but lower wait time to improve patient flow."
    
    prompt += "\nPlease explain in 2-3 sentences why this risk level and department were chosen, in simple language suitable for a triage nurse. Focus on the most important factors that led to this decision. If operational load influenced the department choice, mention it briefly."
    
    return prompt


# ============================================================================
# F. Main Entrypoint
# ============================================================================

def predict_risk_and_department(patient: dict, artifacts: Dict[str, Any]) -> dict:
    """
    Full pipeline: rule engine + ML ensemble + LLM prompt generation.
    
    Args:
        patient: raw patient data dict with all feature fields
        artifacts: output from load_artifacts()
    
    Returns:
        dict with:
            - risk_level: str
            - risk_source: "rules" | "ensemble"
            - risk_probabilities: Dict[str, float] | None
            - department: str
            - department_source: "rules" | "model"
            - rule_reasons: List[str]
            - top_features: List[Tuple[str, float]]
            - llm_prompt: str (prompt to send to LLM externally)
    """
    # Step 1: Apply rule engine
    rule_result = apply_rules(patient)
    
    # Step 2: If forced by rules, return rule-based decision
    if rule_result["forced"]:
        # Evaluate comorbidities for rule-based decisions too
        pre_existing_conditions = patient.get('Pre_Existing_Conditions', '')
        chief_complaint = patient.get('Chief_Complaint', '')
        comorbidities = parse_comorbidities(str(pre_existing_conditions))
        comorbidity_effects_list = evaluate_comorbidities(comorbidities, chief_complaint)
        
        # Convert to dicts
        from comorbidity import COMORBIDITY_RULES
        comorbidity_effects_dicts = []
        complaint_lower = chief_complaint.lower()
        for comorbidity in comorbidities:
            for (rule_comorbidity, complaint_keyword), effect in COMORBIDITY_RULES.items():
                if (rule_comorbidity == comorbidity and 
                    complaint_keyword.lower() in complaint_lower):
                    comorbidity_effects_dicts.append({
                        "comorbidity": comorbidity,
                        "extra_risk_points": effect.extra_risk_points,
                        "preferred_department": effect.preferred_department,
                        "explanation": effect.explanation
                    })
        
        # Build LLM prompt with rule reasons but no ML probabilities
        # For rule-based decisions, skip operational adjustment
        llm_prompt = build_llm_explanation_prompt(
            patient=patient,
            risk_level=rule_result["risk_level"],
            risk_probabilities=None,
            department=rule_result["department"],
            rule_reasons=rule_result["rule_reasons"],
            top_features=[],
            dept_clinical_probs=None,
            dept_operational_scores=None,
            dept_load=None,
            comorbidity_effects=comorbidity_effects_dicts,
            deterioration_info=None
        )
        
        return {
            "risk_level": rule_result["risk_level"],
            "risk_source": "rules",
            "risk_probabilities": None,
            "department": rule_result["department"],
            "department_source": "rules",
            "rule_reasons": rule_result["rule_reasons"],
            "comorbidity_effects": comorbidity_effects_dicts,
            "top_features": [],
            "llm_prompt": llm_prompt
        }
    
    # Step 3: Run ML ensemble prediction
    ensemble_result = predict_with_ensemble(patient, artifacts)
    
    # Step 3.5: Evaluate comorbidities for this patient
    pre_existing_conditions = patient.get('Pre_Existing_Conditions', '')
    chief_complaint = patient.get('Chief_Complaint', '')
    comorbidities = parse_comorbidities(str(pre_existing_conditions))
    comorbidity_effects_list = evaluate_comorbidities(comorbidities, chief_complaint)
    
    # Convert ComorbidityEffect NamedTuples to dicts for JSON serialization
    from comorbidity import COMORBIDITY_RULES
    comorbidity_effects_dicts = []
    complaint_lower = chief_complaint.lower()
    for comorbidity in comorbidities:
        for (rule_comorbidity, complaint_keyword), effect in COMORBIDITY_RULES.items():
            if (rule_comorbidity == comorbidity and 
                complaint_keyword.lower() in complaint_lower):
                comorbidity_effects_dicts.append({
                    "comorbidity": comorbidity,
                    "extra_risk_points": effect.extra_risk_points,
                    "preferred_department": effect.preferred_department,
                    "explanation": effect.explanation
                })
    
    # Step 4: Apply operational awareness to department selection
    # Get current operational load
    current_load = get_current_load()
    
    # Get clinical department probabilities
    dept_clinical_probs = ensemble_result["department_clinical_probs"]
    
    # Check if comorbidity suggests a preferred department
    comorbidity_preferred_dept = get_comorbidity_preferred_department(comorbidities, chief_complaint)
    
    # Choose department considering both clinical and operational factors
    chosen_department, combined_scores = choose_operationally_aware_department(
        clinical_probs=dept_clinical_probs,
        current_load=current_load,
        risk_level=ensemble_result["risk_level"],
        alpha=0.7  # 70% clinical, 30% operational
    )
    
    # Override with comorbidity preference if:
    # 1. Risk is Medium or High
    # 2. Comorbidity preferred department is specified
    # 3. Clinical probability for comorbidity dept is reasonable (>= 0.2)
    if (comorbidity_preferred_dept and 
        ensemble_result["risk_level"] in ["Medium", "High"] and
        dept_clinical_probs.get(comorbidity_preferred_dept, 0) >= 0.2):
        chosen_department = comorbidity_preferred_dept
    
    # Step 5: Build LLM prompt with ML results, operational info, and comorbidity info
    llm_prompt = build_llm_explanation_prompt(
        patient=patient,
        risk_level=ensemble_result["risk_level"],
        risk_probabilities=ensemble_result["risk_probabilities"],
        department=chosen_department,
        rule_reasons=[],  # No rule reasons since ML was used
        top_features=ensemble_result["top_features"],
        dept_clinical_probs=dept_clinical_probs,
        dept_operational_scores=combined_scores,
        dept_load=current_load,
        comorbidity_effects=comorbidity_effects_dicts,
        deterioration_info=None  # Temporal info added separately in monitoring endpoint
    )
    
    return {
        "risk_level": ensemble_result["risk_level"],
        "risk_source": "ensemble",
        "risk_probabilities": ensemble_result["risk_probabilities"],
        "department": chosen_department,
        "department_source": "model",
        "department_clinical_probs": dept_clinical_probs,
        "department_operational_scores": combined_scores,
        "department_load": current_load,
        "comorbidity_effects": comorbidity_effects_dicts,
        "rule_reasons": [],
        "top_features": ensemble_result["top_features"],
        "llm_prompt": llm_prompt
    }


# ============================================================================
# Example Usage
# ============================================================================

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
        'Chief_Complaint': 'chest pain',
        'Symptoms': 'mild discomfort',
        'Allergies': 'None',
        'Current_Medications': 'Aspirin',
        'Pre_Existing_Conditions': 'Hypertension'
    }
    
    try:
        # Load artifacts
        print("Loading models...")
        artifacts = load_artifacts()
        print("✓ Models loaded")
        
        # Test rule engine
        print("\nTesting rule engine...")
        rule_result = apply_rules(example_patient)
        print(f"Rule result: {rule_result}")
        
        # Full prediction
        print("\nRunning full prediction pipeline...")
        result = predict_risk_and_department(example_patient, artifacts)
        
        print(f"\nPrediction Result:")
        print(f"  Risk Level: {result['risk_level']} (source: {result['risk_source']})")
        print(f"  Department: {result['department']} (source: {result['department_source']})")
        if result['risk_probabilities']:
            print(f"  Risk Probabilities: {result['risk_probabilities']}")
        if result['rule_reasons']:
            print(f"  Rule Reasons: {result['rule_reasons']}")
        print(f"\nLLM Prompt (first 200 chars):")
        print(result['llm_prompt'][:200] + "...")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train_triage_models.py first to train and save the models.")
