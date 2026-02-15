"""
Active Questioning Triage Module
Implements a progressive questioning system that asks the most informative questions
based on feature importance until confidence threshold is reached or max questions asked.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import uuid
from triage_inference import (
    apply_rules,
    predict_risk_and_department,
    ALL_FEATURES
)


# ============================================================================
# Question Configuration
# ============================================================================

QUESTION_CONFIG = {
    "Systolic_BP": {
        "question": "What is the patient's systolic blood pressure (mmHg)?",
        "type": "numeric",
        "required": False
    },
    "Heart_Rate": {
        "question": "What is the patient's heart rate (beats per minute)?",
        "type": "numeric",
        "required": False
    },
    "Body_Temperature_C": {
        "question": "What is the patient's body temperature (°C)?",
        "type": "numeric",
        "required": False
    },
    "Respiratory_Rate": {
        "question": "What is the patient's respiratory rate (breaths per minute)?",
        "type": "numeric",
        "required": False
    },
    "SpO2": {
        "question": "What is the patient's oxygen saturation (SpO₂ %)?",
        "type": "numeric",
        "required": False
    },
    "Onset_Duration_Value": {
        "question": "How long has the patient had these symptoms (numeric value)?",
        "type": "numeric",
        "required": False
    },
    "Onset_Duration_Unit": {
        "question": "What is the duration unit? (hours/days/months/years)",
        "type": "categorical",
        "required": False
    },
    "Symptoms": {
        "question": "What are the patient's symptoms? (describe in detail)",
        "type": "text",
        "required": False
    },
    "Allergies": {
        "question": "Does the patient have any known allergies?",
        "type": "text",
        "required": False
    },
    "Current_Medications": {
        "question": "What medications is the patient currently taking?",
        "type": "text",
        "required": False
    },
    "Pre_Existing_Conditions": {
        "question": "Does the patient have any pre-existing conditions? (e.g., Diabetes, Hypertension, COPD, Asthma)",
        "type": "text",
        "required": False
    }
}

# Features that are collected at the start (not asked via active questioning)
INITIAL_FEATURES = ['Age', 'Gender', 'Chief_Complaint']

# Default values for missing features (used when building temporary patient dict)
DEFAULT_VALUES = {
    'Systolic_BP': 120.0,
    'Heart_Rate': 75.0,
    'Body_Temperature_C': 37.0,
    'Respiratory_Rate': 18.0,
    'SpO2': 98.0,
    'Onset_Duration_Value': 1.0,
    'Onset_Duration_Unit': 'hours',
    'Symptoms': 'Not specified',
    'Allergies': 'None',
    'Current_Medications': 'None',
    'Pre_Existing_Conditions': 'None'
}


# ============================================================================
# Global Feature Importance Computation
# ============================================================================

def compute_global_feature_importance(artifacts: Dict[str, Any], 
                                      sample_size: int = 1000) -> List[str]:
    """
    Compute global feature importance using SHAP values on a sample of data.
    
    This is a deterministic, greedy approach (NOT reinforcement learning):
    - Uses mean absolute SHAP values across a sample to rank features
    - Higher importance = ask this question earlier
    
    Args:
        artifacts: loaded models and preprocessor
        sample_size: number of samples to use for importance computation
    
    Returns:
        List of feature names sorted by importance (descending)
    """
    try:
        shap_explainer = artifacts.get("shap_explainer")
        preprocessor = artifacts.get("preprocessor")
        risk_xgb = artifacts.get("risk_xgb")
        
        if shap_explainer is None or preprocessor is None or risk_xgb is None:
            # Fallback: use default importance order
            return get_default_importance_ranking()
        
        # Generate synthetic samples for importance computation
        # We'll create diverse samples based on typical ranges
        np.random.seed(42)  # Deterministic
        
        samples = []
        for _ in range(sample_size):
            sample = {
                'Age': np.random.uniform(1, 95),
                'Gender': np.random.choice(['M', 'F', 'T']),
                'Chief_Complaint': np.random.choice([
                    'chest pain', 'shortness of breath', 'abdominal pain',
                    'dizziness', 'fever and cough', 'severe headache'
                ]),
                'Systolic_BP': np.random.uniform(70, 200),
                'Heart_Rate': np.random.uniform(40, 150),
                'Body_Temperature_C': np.random.uniform(35, 40),
                'Respiratory_Rate': np.random.uniform(8, 30),
                'SpO2': np.random.uniform(85, 100),
                'Onset_Duration_Value': np.random.uniform(0.5, 24),
                'Onset_Duration_Unit': np.random.choice(['hours', 'days', 'months', 'years']),
                'Symptoms': 'Not specified',
                'Allergies': 'None',
                'Current_Medications': 'None',
                'Pre_Existing_Conditions': 'None'
            }
            samples.append(sample)
        
        # Convert to DataFrame
        sample_df = pd.DataFrame(samples)
        
        # Transform using preprocessor
        X_transformed = preprocessor.transform(sample_df)
        
        # Compute SHAP values
        shap_values = shap_explainer.shap_values(X_transformed)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
            # Use mean absolute SHAP across classes
            shap_values = np.mean(np.abs(shap_values), axis=0)
        else:
            shap_values = np.abs(shap_values)
        
        # Get feature names
        feature_names = []
        
        # Numeric features
        numeric_transformer = preprocessor.transformers_[0]
        if numeric_transformer[0] == 'num':
            numeric_features = numeric_transformer[2]
            feature_names.extend(numeric_features)
        
        # Categorical features
        cat_transformer = preprocessor.transformers_[1]
        if cat_transformer[0] == 'cat':
            cat_columns = cat_transformer[2]
            cat_encoder = preprocessor.named_transformers_['cat']
            
            if hasattr(cat_encoder, 'get_feature_names_out'):
                cat_feature_names = cat_encoder.get_feature_names_out(cat_columns)
                feature_names.extend(cat_feature_names)
            elif hasattr(cat_encoder, 'categories_'):
                for i, col in enumerate(cat_columns):
                    if i < len(cat_encoder.categories_):
                        for cat in cat_encoder.categories_[i]:
                            feature_names.append(f"{col}_{cat}")
        
        # Compute mean absolute SHAP per original feature
        # Map transformed features back to original features
        original_feature_importance = {}
        
        # Ensure shap_values is 2D (samples x features)
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)
        
        # For numeric features, direct mapping
        for i, feat in enumerate(numeric_features):
            if i < shap_values.shape[1]:
                original_feature_importance[feat] = np.mean(np.abs(shap_values[:, i]))
        
        # For categorical features, aggregate by original column
        cat_start_idx = len(numeric_features)
        for col in cat_columns:
            col_importance = 0.0
            col_count = 0
            for i, feat_name in enumerate(feature_names):
                if feat_name.startswith(f"{col}_") and i >= cat_start_idx:
                    idx = i - cat_start_idx
                    if idx < shap_values.shape[1]:
                        col_importance += np.mean(np.abs(shap_values[:, idx]))
                        col_count += 1
            if col_count > 0:
                original_feature_importance[col] = col_importance / col_count
        
        # Sort by importance (descending)
        sorted_features = sorted(
            original_feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return feature names only, filtered to QUESTION_CONFIG features
        importance_ranking = [
            feat for feat, _ in sorted_features
            if feat in QUESTION_CONFIG
        ]
        
        # Add any missing QUESTION_CONFIG features at the end
        for feat in QUESTION_CONFIG:
            if feat not in importance_ranking:
                importance_ranking.append(feat)
        
        return importance_ranking
    
    except Exception as e:
        # Fallback to default ranking
        return get_default_importance_ranking()


def get_default_importance_ranking() -> List[str]:
    """
    Default feature importance ranking (fallback if SHAP computation fails).
    Based on clinical importance for triage.
    """
    return [
        'Systolic_BP',
        'SpO2',
        'Heart_Rate',
        'Respiratory_Rate',
        'Body_Temperature_C',
        'Symptoms',
        'Pre_Existing_Conditions',
        'Onset_Duration_Value',
        'Onset_Duration_Unit',
        'Current_Medications',
        'Allergies'
    ]


# ============================================================================
# Session Management
# ============================================================================

def initialize_session(initial_answers: Dict[str, Any],
                       max_questions: int = 10) -> Dict[str, Any]:
    """
    Initialize a new active questioning session.
    
    Args:
        initial_answers: minimal known info (e.g., Age, Gender, Chief_Complaint, Symptoms)
        max_questions: maximum number of questions to ask
    
    Returns:
        Session dict with:
            - session_id: unique identifier
            - known_answers: dict of answered questions
            - asked_questions: list of feature names already asked
            - max_questions: maximum questions allowed
            - confidence_threshold: threshold for stopping (default 0.75)
    """
    session_id = str(uuid.uuid4())
    
    # Validate initial answers contain required fields
    for feat in INITIAL_FEATURES:
        if feat not in initial_answers:
            raise ValueError(f"Missing required initial field: {feat}")
    
    session = {
        "session_id": session_id,
        "known_answers": initial_answers.copy(),
        "asked_questions": [],
        "max_questions": max_questions,
        "confidence_threshold": 0.75
    }
    
    return session


def update_session_with_answer(session: Dict[str, Any],
                               feature: str,
                               value: Any) -> Dict[str, Any]:
    """
    Update session with a new answer.
    
    Args:
        session: current session state
        feature: feature name that was answered
        value: answer value
    
    Returns:
        Updated session dict
    """
    if feature not in QUESTION_CONFIG:
        raise ValueError(f"Feature '{feature}' is not in QUESTION_CONFIG")
    
    session["known_answers"][feature] = value
    
    if feature not in session["asked_questions"]:
        session["asked_questions"].append(feature)
    
    return session


# ============================================================================
# Question Selection Logic
# ============================================================================

def build_patient_dict_from_session(session: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a complete patient dict from session answers, using defaults for missing values.
    
    Args:
        session: current session state
    
    Returns:
        Complete patient dict ready for prediction
    """
    patient = session["known_answers"].copy()
    
    # Fill in defaults for missing features
    for feat in ALL_FEATURES:
        if feat not in patient:
            if feat in DEFAULT_VALUES:
                patient[feat] = DEFAULT_VALUES[feat]
            else:
                # For features not in defaults, use None (will be handled by preprocessing)
                patient[feat] = None
    
    return patient


def select_next_question(session: Dict[str, Any],
                        artifacts: Dict[str, Any],
                        importance_ranking: List[str],
                        confidence_threshold: float = 0.75) -> Dict[str, Any]:
    """
    Select the next question to ask or determine if questioning is complete.
    
    This implements a deterministic, greedy questioning strategy (NOT reinforcement learning):
    - Asks questions in order of feature importance
    - Stops when confidence threshold reached or max questions asked
    - Uses existing rule engine and ML ensemble for predictions
    
    Args:
        session: current triage session state
        artifacts: loaded models/preprocessor
        importance_ranking: features sorted by global importance
        confidence_threshold: probability threshold to stop asking (default 0.75)
    
    Returns:
        dict with:
            - done: bool (whether questioning is complete)
            - question_feature: Optional[str] (next feature to ask about)
            - question_text: Optional[str] (human-readable question)
            - current_risk_prediction: Optional[str] (current risk level)
            - current_risk_probabilities: Optional[Dict[str, float]] (current probabilities)
            - current_department: Optional[str] (current department assignment)
            - rule_triggered: bool (whether a rule-based decision was made)
    """
    # Build patient dict with current answers + defaults
    patient = build_patient_dict_from_session(session)
    
    # Check rules first (rules can override and stop questioning immediately)
    rule_result = apply_rules(patient)
    
    if rule_result["forced"]:
        # Rule triggered - stop questioning and return final decision
        return {
            "done": True,
            "question_feature": None,
            "question_text": None,
            "current_risk_prediction": rule_result["risk_level"],
            "current_risk_probabilities": None,
            "current_department": rule_result["department"],
            "rule_triggered": True,
            "rule_reasons": rule_result["rule_reasons"]
        }
    
    # Run ML prediction with current information
    try:
        prediction_result = predict_risk_and_department(patient, artifacts)
        
        risk_level = prediction_result["risk_level"]
        risk_probabilities = prediction_result["risk_probabilities"]
        department = prediction_result["department"]
        
        # Check if we should stop questioning
        max_prob = max(risk_probabilities.values()) if risk_probabilities else 0.0
        questions_asked = len(session["asked_questions"])
        max_questions = session.get("max_questions", 10)
        
        should_stop = (
            max_prob >= confidence_threshold or
            questions_asked >= max_questions
        )
        
        if should_stop:
            # Stop questioning - return final decision
            return {
                "done": True,
                "question_feature": None,
                "question_text": None,
                "current_risk_prediction": risk_level,
                "current_risk_probabilities": risk_probabilities,
                "current_department": department,
                "rule_triggered": False,
                "confidence": max_prob,
                "questions_asked": questions_asked
            }
        
        # Find next question to ask
        # Pick highest importance feature that:
        # 1. Is in QUESTION_CONFIG
        # 2. Is not yet in known_answers
        # 3. Is not yet in asked_questions
        next_feature = None
        for feat in importance_ranking:
            if (feat in QUESTION_CONFIG and
                feat not in session["known_answers"] and
                feat not in session["asked_questions"]):
                next_feature = feat
                break
        
        if next_feature is None:
            # No more questions available - return current prediction
            return {
                "done": True,
                "question_feature": None,
                "question_text": None,
                "current_risk_prediction": risk_level,
                "current_risk_probabilities": risk_probabilities,
                "current_department": department,
                "rule_triggered": False,
                "confidence": max_prob,
                "questions_asked": questions_asked,
                "reason": "No more questions available"
            }
        
        # Return next question
        question_text = QUESTION_CONFIG[next_feature]["question"]
        
        return {
            "done": False,
            "question_feature": next_feature,
            "question_text": question_text,
            "question_type": QUESTION_CONFIG[next_feature]["type"],
            "current_risk_prediction": risk_level,
            "current_risk_probabilities": risk_probabilities,
            "current_department": department,
            "rule_triggered": False,
            "confidence": max_prob,
            "questions_asked": questions_asked
        }
    
    except Exception as e:
        # If prediction fails, return error state
        return {
            "done": True,
            "question_feature": None,
            "question_text": None,
            "current_risk_prediction": None,
            "current_risk_probabilities": None,
            "current_department": None,
            "rule_triggered": False,
            "error": str(e)
        }


# ============================================================================
# Session Storage (In-Memory)
# ============================================================================

# Global session storage (in production, use Redis or database)
_sessions: Dict[str, Dict[str, Any]] = {}


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session by ID."""
    return _sessions.get(session_id)


def save_session(session: Dict[str, Any]) -> None:
    """Save session to storage."""
    _sessions[session["session_id"]] = session


def delete_session(session_id: str) -> None:
    """Delete session from storage."""
    if session_id in _sessions:
        del _sessions[session_id]
