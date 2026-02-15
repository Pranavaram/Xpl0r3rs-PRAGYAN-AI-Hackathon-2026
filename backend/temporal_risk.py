"""
Temporal Risk Evolution Module
Tracks patient risk over multiple timepoints and flags deterioration.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from typing_extensions import TypedDict


# ============================================================================
# Data Structures
# ============================================================================

class RiskSnapshot(TypedDict):
    """A single risk assessment snapshot at a point in time."""
    timestamp: datetime
    risk_level: str
    risk_probabilities: Dict[str, float]
    department: str


class RiskTimeline(TypedDict):
    """Complete risk timeline for a patient."""
    patient_id: str
    snapshots: List[RiskSnapshot]


# ============================================================================
# Timeline Management
# ============================================================================

def add_snapshot(
    timeline: RiskTimeline,
    risk_level: str,
    risk_probabilities: Dict[str, float],
    department: str,
    timestamp: Optional[datetime] = None
) -> RiskTimeline:
    """
    Append a new RiskSnapshot with the given data and return the updated timeline.
    
    Args:
        timeline: Existing risk timeline
        risk_level: Current risk level (Low/Medium/High)
        risk_probabilities: Risk probabilities dict
        department: Assigned department
        timestamp: Timestamp for snapshot (defaults to now if None)
    
    Returns:
        Updated timeline with new snapshot appended
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    snapshot: RiskSnapshot = {
        "timestamp": timestamp,
        "risk_level": risk_level,
        "risk_probabilities": risk_probabilities.copy() if risk_probabilities else {},
        "department": department
    }
    
    # Create a new timeline dict to avoid mutating the original
    updated_timeline: RiskTimeline = {
        "patient_id": timeline["patient_id"],
        "snapshots": timeline["snapshots"].copy() + [snapshot]
    }
    
    return updated_timeline


def compute_deterioration_flags(timeline: RiskTimeline) -> Dict[str, Any]:
    """
    Analyze the snapshots and detect deterioration trends.
    
    Args:
        timeline: Risk timeline with snapshots
    
    Returns:
        Dict with:
            - is_deteriorating: bool
            - reason: str (explanation of deterioration if detected)
            - last_risk_level: str
            - previous_risk_level: Optional[str]
            - trend: List[Dict] with timestamp and risk_level
    """
    snapshots = timeline["snapshots"]
    
    if len(snapshots) == 0:
        return {
            "is_deteriorating": False,
            "reason": "No previous readings available",
            "last_risk_level": None,
            "previous_risk_level": None,
            "trend": []
        }
    
    # Consider only the last 3 snapshots for trend analysis
    recent_snapshots = snapshots[-3:] if len(snapshots) > 3 else snapshots
    
    # Build trend list
    trend = [
        {
            "timestamp": snap["timestamp"].isoformat(),
            "risk_level": snap["risk_level"]
        }
        for snap in recent_snapshots
    ]
    
    last_snapshot = recent_snapshots[-1]
    last_risk_level = last_snapshot["risk_level"]
    last_probabilities = last_snapshot.get("risk_probabilities", {})
    
    previous_risk_level = None
    if len(recent_snapshots) > 1:
        previous_snapshot = recent_snapshots[-2]
        previous_risk_level = previous_snapshot["risk_level"]
        previous_probabilities = previous_snapshot.get("risk_probabilities", {})
    
    # Check for deterioration
    is_deteriorating = False
    reason = "Risk level stable"
    
    # Rule 1: Risk level upgrade (Low → Medium, Medium → High, Low → High)
    if previous_risk_level is not None:
        risk_hierarchy = {"Low": 0, "Medium": 1, "High": 2}
        last_level_num = risk_hierarchy.get(last_risk_level, 0)
        prev_level_num = risk_hierarchy.get(previous_risk_level, 0)
        
        if last_level_num > prev_level_num:
            is_deteriorating = True
            reason = f"Risk increased from {previous_risk_level} to {last_risk_level} in the latest reading."
    
    # Rule 2: High-risk probability increased significantly
    if not is_deteriorating and previous_probabilities and last_probabilities:
        prev_high_prob = previous_probabilities.get("High", 0.0)
        last_high_prob = last_probabilities.get("High", 0.0)
        
        if last_high_prob - prev_high_prob > 0.2:
            is_deteriorating = True
            reason = f"High-risk probability increased from {prev_high_prob:.1%} to {last_high_prob:.1%} over recent readings."
    
    # Rule 3: Medium-risk probability increased significantly while High also increased
    if not is_deteriorating and previous_probabilities and last_probabilities:
        prev_medium_prob = previous_probabilities.get("Medium", 0.0)
        last_medium_prob = last_probabilities.get("Medium", 0.0)
        prev_high_prob = previous_probabilities.get("High", 0.0)
        last_high_prob = last_probabilities.get("High", 0.0)
        
        if (last_medium_prob - prev_medium_prob > 0.15 and 
            last_high_prob > prev_high_prob and
            last_risk_level == "Medium"):
            is_deteriorating = True
            reason = f"Risk probabilities shifting toward higher risk (Medium: {prev_medium_prob:.1%} → {last_medium_prob:.1%}, High: {prev_high_prob:.1%} → {last_high_prob:.1%})."
    
    # Rule 4: Check if we have 3+ snapshots and see overall trend
    if not is_deteriorating and len(recent_snapshots) >= 3:
        # Check if risk level has been consistently increasing
        risk_levels = [snap["risk_level"] for snap in recent_snapshots]
        risk_nums = [risk_hierarchy.get(level, 0) for level in risk_levels]
        
        # If all three are increasing
        if (risk_nums[0] < risk_nums[1] < risk_nums[2]):
            is_deteriorating = True
            reason = f"Progressive risk increase observed: {' → '.join(risk_levels)}"
    
    return {
        "is_deteriorating": is_deteriorating,
        "reason": reason,
        "last_risk_level": last_risk_level,
        "previous_risk_level": previous_risk_level,
        "trend": trend
    }


# ============================================================================
# Integration with Prediction Pipeline
# ============================================================================

def update_timeline_with_prediction(
    timeline: RiskTimeline,
    patient: dict,
    artifacts: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update timeline with a new prediction and compute deterioration flags.
    
    This function:
    1. Calls predict_risk_and_department(patient, artifacts)
    2. Adds the new snapshot to the timeline
    3. Computes deterioration flags
    4. Returns combined result
    
    Args:
        timeline: Existing risk timeline (or new timeline if first reading)
        patient: Patient data dict
        artifacts: Loaded ML models and preprocessor
    
    Returns:
        Dict with:
            - timeline: updated timeline
            - prediction: full result from predict_risk_and_department
            - deterioration: deterioration flags and info
    """
    from triage_inference import predict_risk_and_department
    
    # Get prediction from existing pipeline
    prediction_result = predict_risk_and_department(patient, artifacts)
    
    # Extract key fields for snapshot
    risk_level = prediction_result["risk_level"]
    risk_probabilities = prediction_result.get("risk_probabilities", {})
    department = prediction_result["department"]
    
    # Add snapshot to timeline
    updated_timeline = add_snapshot(
        timeline=timeline,
        risk_level=risk_level,
        risk_probabilities=risk_probabilities,
        department=department
    )
    
    # Compute deterioration flags
    deterioration_info = compute_deterioration_flags(updated_timeline)
    
    return {
        "timeline": updated_timeline,
        "prediction": prediction_result,
        "deterioration": deterioration_info
    }


# ============================================================================
# Timeline Storage (In-Memory)
# ============================================================================

# Global in-memory storage for timelines
# In production, this would be Redis or a database
_timelines: Dict[str, RiskTimeline] = {}


def get_timeline(patient_id: str) -> Optional[RiskTimeline]:
    """Get timeline for a patient, or None if not found."""
    return _timelines.get(patient_id)


def create_timeline(patient_id: str) -> RiskTimeline:
    """Create a new empty timeline for a patient."""
    timeline: RiskTimeline = {
        "patient_id": patient_id,
        "snapshots": []
    }
    _timelines[patient_id] = timeline
    return timeline


def get_or_create_timeline(patient_id: str) -> RiskTimeline:
    """Get existing timeline or create a new one."""
    timeline = get_timeline(patient_id)
    if timeline is None:
        timeline = create_timeline(patient_id)
    return timeline


def save_timeline(timeline: RiskTimeline) -> None:
    """Save timeline to storage."""
    _timelines[timeline["patient_id"]] = timeline


def delete_timeline(patient_id: str) -> None:
    """Delete timeline from storage."""
    if patient_id in _timelines:
        del _timelines[patient_id]
