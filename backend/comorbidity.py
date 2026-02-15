"""
Comorbidity Module for Triage System
Handles structured comorbidity representation and interaction rules.
"""

from typing import List, Dict, NamedTuple, Optional


# ============================================================================
# Comorbidity Codes
# ============================================================================

COMORBIDITY_CODES = [
    "Diabetes",
    "Hypertension",
    "Asthma",
    "COPD",
    "CAD",  # Coronary Artery Disease
    "CKD",  # Chronic Kidney Disease
    "Heart Failure"
]

# Mapping from common variations/synonyms to standardized codes
COMORBIDITY_SYNONYMS: Dict[str, str] = {
    # Diabetes variations
    "diabetes": "Diabetes",
    "diabetes mellitus": "Diabetes",
    "type 2 diabetes": "Diabetes",
    "type 1 diabetes": "Diabetes",
    "diabetic": "Diabetes",
    
    # Hypertension variations
    "hypertension": "Hypertension",
    "high blood pressure": "Hypertension",
    "htn": "Hypertension",
    "elevated bp": "Hypertension",
    
    # Asthma variations
    "asthma": "Asthma",
    "asthmatic": "Asthma",
    
    # COPD variations
    "copd": "COPD",
    "chronic obstructive pulmonary disease": "COPD",
    "emphysema": "COPD",
    "chronic bronchitis": "COPD",
    
    # CAD variations
    "cad": "CAD",
    "coronary artery disease": "CAD",
    "coronary heart disease": "CAD",
    "chd": "CAD",
    "ischemic heart disease": "CAD",
    "ihd": "CAD",
    
    # CKD variations
    "ckd": "CKD",
    "chronic kidney disease": "CKD",
    "renal failure": "CKD",
    "kidney disease": "CKD",
    "ckf": "CKD",  # Chronic Kidney Failure
    
    # Heart Failure variations
    "heart failure": "Heart Failure",
    "hf": "Heart Failure",
    "congestive heart failure": "Heart Failure",
    "chf": "Heart Failure",
    "cardiac failure": "Heart Failure"
}


# ============================================================================
# Comorbidity Effect Structure
# ============================================================================

class ComorbidityEffect(NamedTuple):
    """Represents the effect of a comorbidity interaction."""
    extra_risk_points: int
    preferred_department: Optional[str]
    explanation: str


# ============================================================================
# Comorbidity Interaction Rules
# ============================================================================

# Keys are (comorbidity, complaint_keyword)
# complaint_keyword is matched via substring search (case-insensitive)
COMORBIDITY_RULES: Dict[tuple[str, str], ComorbidityEffect] = {
    # CAD (Coronary Artery Disease) interactions
    ("CAD", "chest pain"): ComorbidityEffect(
        extra_risk_points=3,
        preferred_department="Cardiology",
        explanation="Coronary artery disease with chest pain increases risk of acute coronary syndrome."
    ),
    ("CAD", "shortness of breath"): ComorbidityEffect(
        extra_risk_points=2,
        preferred_department="Cardiology",
        explanation="CAD with shortness of breath may indicate cardiac decompensation."
    ),
    ("CAD", "dizziness"): ComorbidityEffect(
        extra_risk_points=2,
        preferred_department="Cardiology",
        explanation="CAD with dizziness may indicate arrhythmia or cardiac event."
    ),
    
    # COPD interactions
    ("COPD", "shortness of breath"): ComorbidityEffect(
        extra_risk_points=2,
        preferred_department="Pulmonology",
        explanation="COPD with shortness of breath increases risk of respiratory decompensation."
    ),
    ("COPD", "fever"): ComorbidityEffect(
        extra_risk_points=2,
        preferred_department="Pulmonology",
        explanation="COPD with fever may indicate pneumonia or respiratory infection requiring urgent care."
    ),
    ("COPD", "cough"): ComorbidityEffect(
        extra_risk_points=1,
        preferred_department="Pulmonology",
        explanation="COPD with cough may indicate exacerbation."
    ),
    
    # Asthma interactions
    ("Asthma", "shortness of breath"): ComorbidityEffect(
        extra_risk_points=2,
        preferred_department="Pulmonology",
        explanation="Asthma with shortness of breath may indicate severe exacerbation requiring urgent treatment."
    ),
    ("Asthma", "wheezing"): ComorbidityEffect(
        extra_risk_points=1,
        preferred_department="Pulmonology",
        explanation="Asthma with wheezing suggests active bronchospasm."
    ),
    
    # Diabetes interactions
    ("Diabetes", "fever"): ComorbidityEffect(
        extra_risk_points=1,
        preferred_department=None,
        explanation="Diabetes with infection may progress faster and needs closer monitoring."
    ),
    ("Diabetes", "chest pain"): ComorbidityEffect(
        extra_risk_points=2,
        preferred_department="Cardiology",
        explanation="Diabetic patients with chest pain have higher risk of silent MI and need cardiac evaluation."
    ),
    ("Diabetes", "dizziness"): ComorbidityEffect(
        extra_risk_points=1,
        preferred_department=None,
        explanation="Diabetes with dizziness may indicate hypoglycemia or autonomic dysfunction."
    ),
    
    # Heart Failure interactions
    ("Heart Failure", "shortness of breath"): ComorbidityEffect(
        extra_risk_points=3,
        preferred_department="Cardiology",
        explanation="Heart failure with shortness of breath indicates potential acute decompensation requiring urgent cardiac care."
    ),
    ("Heart Failure", "chest pain"): ComorbidityEffect(
        extra_risk_points=2,
        preferred_department="Cardiology",
        explanation="Heart failure with chest pain may indicate acute coronary event or worsening cardiac function."
    ),
    ("Heart Failure", "swelling"): ComorbidityEffect(
        extra_risk_points=2,
        preferred_department="Cardiology",
        explanation="Heart failure with swelling suggests fluid overload and decompensation."
    ),
    
    # CKD interactions
    ("CKD", "fever"): ComorbidityEffect(
        extra_risk_points=1,
        preferred_department=None,
        explanation="CKD with infection may require adjusted antibiotic dosing and closer monitoring."
    ),
    ("CKD", "chest pain"): ComorbidityEffect(
        extra_risk_points=2,
        preferred_department="Cardiology",
        explanation="CKD patients have higher cardiovascular risk and need careful cardiac evaluation."
    ),
    
    # Hypertension interactions
    ("Hypertension", "chest pain"): ComorbidityEffect(
        extra_risk_points=1,
        preferred_department="Cardiology",
        explanation="Hypertension with chest pain increases concern for cardiac event."
    ),
    ("Hypertension", "severe headache"): ComorbidityEffect(
        extra_risk_points=2,
        preferred_department="Emergency",
        explanation="Hypertension with severe headache may indicate hypertensive emergency."
    )
}


# ============================================================================
# Parsing Functions
# ============================================================================

def parse_comorbidities(raw: str) -> List[str]:
    """
    Parse and normalize comorbidity string into a list of standardized codes.
    
    Args:
        raw: Comma-separated string from Pre_Existing_Conditions field
             (e.g., "Diabetes, Hypertension, CAD" or "None")
    
    Returns:
        List of normalized comorbidity codes (subset of COMORBIDITY_CODES)
    """
    if not raw or raw.strip().lower() in ["none", "n/a", "na", ""]:
        return []
    
    # Split by comma and clean
    raw_list = [item.strip() for item in raw.split(",")]
    
    # Normalize and map to codes
    normalized = []
    for item in raw_list:
        if not item:
            continue
        
        # Convert to lowercase for matching
        item_lower = item.lower().strip()
        
        # Check synonyms first
        if item_lower in COMORBIDITY_SYNONYMS:
            code = COMORBIDITY_SYNONYMS[item_lower]
            if code not in normalized:
                normalized.append(code)
        else:
            # Try direct match (case-insensitive)
            for code in COMORBIDITY_CODES:
                if code.lower() == item_lower:
                    if code not in normalized:
                        normalized.append(code)
                    break
    
    return normalized


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_comorbidities(comorbidities: List[str], chief_complaint: str) -> List[ComorbidityEffect]:
    """
    Evaluate comorbidity interactions with chief complaint.
    
    Args:
        comorbidities: List of normalized comorbidity codes
        chief_complaint: Chief complaint string (e.g., "chest pain", "shortness of breath")
    
    Returns:
        List of ComorbidityEffect objects for matching interactions
    """
    if not comorbidities or not chief_complaint:
        return []
    
    effects = []
    complaint_lower = chief_complaint.lower()
    
    # Check each comorbidity against complaint keywords
    for comorbidity in comorbidities:
        # Search for matching rules
        for (rule_comorbidity, complaint_keyword), effect in COMORBIDITY_RULES.items():
            if (rule_comorbidity == comorbidity and 
                complaint_keyword.lower() in complaint_lower):
                effects.append(effect)
    
    return effects


def get_comorbidity_risk_adjustment(comorbidities: List[str], chief_complaint: str) -> int:
    """
    Get total risk point adjustment from all comorbidity interactions.
    
    Args:
        comorbidities: List of normalized comorbidity codes
        chief_complaint: Chief complaint string
    
    Returns:
        Total extra risk points (sum of all matching effects)
    """
    effects = evaluate_comorbidities(comorbidities, chief_complaint)
    return sum(effect.extra_risk_points for effect in effects)


def get_comorbidity_preferred_department(comorbidities: List[str], chief_complaint: str) -> Optional[str]:
    """
    Get preferred department based on comorbidity interactions.
    Returns the first matching preferred_department, or None if no match.
    
    Args:
        comorbidities: List of normalized comorbidity codes
        chief_complaint: Chief complaint string
    
    Returns:
        Preferred department name or None
    """
    effects = evaluate_comorbidities(comorbidities, chief_complaint)
    
    # Return first non-None preferred department
    for effect in effects:
        if effect.preferred_department:
            return effect.preferred_department
    
    return None
