"""
Operations Module for Department Load Management
Tracks and manages operational load (queue length/busyness) per department.
"""

from typing import Dict
import random

# All departments in the system
DEPARTMENTS = [
    "General_Medicine",
    "Cardiology",
    "Neurology",
    "Orthopedics",
    "Emergency",
    "Pulmonology"
]

# In-memory storage for current load (0.0 = no load, 1.0 = fully saturated)
# For now, using simulated/static values
# In production, this would be updated from real-time queue/occupancy data
_current_load: Dict[str, float] = {
    "General_Medicine": 0.3,
    "Cardiology": 0.7,
    "Neurology": 0.4,
    "Orthopedics": 0.5,
    "Emergency": 0.9,
    "Pulmonology": 0.2,
}


def get_current_load() -> Dict[str, float]:
    """
    Get current operational load for all departments.
    
    Returns:
        Dictionary mapping department names to load values (0.0 to 1.0)
    """
    return _current_load.copy()


def update_load(dept: str, new_value: float) -> None:
    """
    Update the load for a specific department.
    
    Args:
        dept: Department name
        new_value: New load value (should be between 0.0 and 1.0)
    
    Raises:
        ValueError: If department not found or value out of range
    """
    if dept not in DEPARTMENTS:
        raise ValueError(f"Unknown department: {dept}")
    
    if not (0.0 <= new_value <= 1.0):
        raise ValueError(f"Load value must be between 0.0 and 1.0, got {new_value}")
    
    _current_load[dept] = new_value


def simulate_load_update() -> None:
    """
    Simulate a load update (for testing/demo purposes).
    Randomly adjusts load values slightly.
    """
    for dept in DEPARTMENTS:
        current = _current_load[dept]
        # Random walk: change by ±0.05, clamped to [0, 1]
        change = random.uniform(-0.05, 0.05)
        new_value = max(0.0, min(1.0, current + change))
        _current_load[dept] = new_value


def get_load_description(load: float) -> str:
    """
    Get a human-readable description of load level.
    
    Args:
        load: Load value (0.0 to 1.0)
    
    Returns:
        Description string
    """
    if load >= 0.9:
        return "very busy"
    elif load >= 0.7:
        return "busy"
    elif load >= 0.5:
        return "moderate"
    elif load >= 0.3:
        return "light"
    else:
        return "available"


def choose_operationally_aware_department(
    clinical_probs: Dict[str, float],
    current_load: Dict[str, float],
    risk_level: str,
    alpha: float = 0.7
) -> tuple[str, Dict[str, float]]:
    """
    Choose department by combining clinical probabilities with operational load.
    
    Safety: For High risk cases, always choose the clinically best department
    regardless of load. For other cases, use combined scoring.
    
    Formula: combined_score = alpha * clinical_prob + (1 - alpha) * (1 - load)
    - Higher clinical probability → higher score
    - Lower load (less busy) → higher score
    - alpha controls weight: 0.7 means 70% clinical, 30% operational
    
    Args:
        clinical_probs: Per-department probabilities from ML model (sums to ~1.0)
        current_load: Per-department load scores in [0, 1]
        risk_level: Patient risk level ("Low", "Medium", "High")
        alpha: Weight for clinical vs operational (default 0.7)
    
    Returns:
        Tuple of (chosen_department, combined_scores_dict)
    """
    # Safety: For High risk, always choose clinically best department
    if risk_level == "High":
        # Find department with highest clinical probability
        chosen_dept = max(clinical_probs.items(), key=lambda x: x[1])[0]
        # Return clinical probabilities as combined scores (no load adjustment)
        return chosen_dept, clinical_probs.copy()
    
    # For Low/Medium risk: combine clinical and operational factors
    combined_scores = {}
    
    for dept, clinical_prob in clinical_probs.items():
        # Get load for this department (default to 0.5 if not found)
        load = current_load.get(dept, 0.5)
        
        # Combined score: alpha * clinical + (1 - alpha) * operational
        # Operational component: (1 - load) so lower load = higher score
        operational_score = 1.0 - load
        combined_score = alpha * clinical_prob + (1 - alpha) * operational_score
        
        combined_scores[dept] = combined_score
    
    # Choose department with highest combined score
    chosen_dept = max(combined_scores.items(), key=lambda x: x[1])[0]
    
    return chosen_dept, combined_scores
