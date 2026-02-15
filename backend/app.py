"""
FastAPI Application for Triage Prediction Service
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Union
import logging
import os
import tempfile
import time
from pathlib import Path

# Try to load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly

from triage_inference import load_artifacts, predict_risk_and_department
from groq_llm import call_llm
from active_questioning import (
    initialize_session,
    update_session_with_answer,
    select_next_question,
    compute_global_feature_importance,
    get_session,
    save_session,
    delete_session,
    QUESTION_CONFIG
)
from temporal_risk import (
    update_timeline_with_prediction,
    get_or_create_timeline,
    get_timeline,
    save_timeline,
    delete_timeline as delete_risk_timeline,
    RiskTimeline
)
from ehr_integration.fhir_parser import FHIRParser
from ehr_integration.document_reader import EHRDocumentReader
from comorbidity import COMORBIDITY_RULES, COMORBIDITY_CODES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Triage Prediction API",
    description="API for medical triage risk assessment and department assignment",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store loaded artifacts (loaded once at startup)
artifacts = None
feature_importance_ranking = None  # Global feature importance ranking for active questioning


# ============================================================================
# Request/Response Models
# ============================================================================

class PatientRequest(BaseModel):
    """Patient data request model."""
    PatientID: Optional[str] = None
    Age: float
    Gender: str
    Chief_Complaint: str
    Symptoms: str
    Systolic_BP: float
    Heart_Rate: float
    Body_Temperature_C: float
    Respiratory_Rate: float
    SpO2: float
    Onset_Duration_Value: float
    Onset_Duration_Unit: str
    Allergies: Optional[str] = None
    Current_Medications: Optional[str] = None
    Pre_Existing_Conditions: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class ComorbidityEffectModel(BaseModel):
    """Comorbidity effect model."""
    comorbidity: str
    extra_risk_points: int
    preferred_department: Optional[str] = None
    explanation: str


class TriageResponse(BaseModel):
    """Triage prediction response model."""
    risk_level: str
    risk_source: str
    risk_probabilities: Optional[Dict[str, float]] = None
    department: str
    department_source: str
    rule_reasons: List[str]
    top_features: List[List] = Field(default_factory=list, description="List of [feature_name, shap_value] pairs from SHAP explainability")
    department_clinical_probs: Optional[Dict[str, float]] = Field(default=None, description="Clinical probabilities per department from ML model")
    department_operational_scores: Optional[Dict[str, float]] = Field(default=None, description="Combined clinical+operational scores per department")
    department_load: Optional[Dict[str, float]] = Field(default=None, description="Current operational load per department (0.0-1.0)")
    comorbidity_effects: List[ComorbidityEffectModel] = Field(default_factory=list, description="Comorbidity interaction effects")
    nl_explanation: str


class SessionStartRequest(BaseModel):
    """Request to start an active questioning session."""
    Age: float
    Gender: str
    Chief_Complaint: str
    Symptoms: Optional[str] = None
    Systolic_BP: Optional[float] = None
    Heart_Rate: Optional[float] = None
    Body_Temperature_C: Optional[float] = None
    Respiratory_Rate: Optional[float] = None
    SpO2: Optional[float] = None
    Onset_Duration_Value: Optional[float] = None
    Onset_Duration_Unit: Optional[str] = None
    Allergies: Optional[str] = None
    Current_Medications: Optional[str] = None
    Pre_Existing_Conditions: Optional[str] = None
    max_questions: Optional[int] = Field(default=10, ge=1, le=15, description="Maximum number of questions to ask")
    confidence_threshold: Optional[float] = Field(default=0.75, ge=0.5, le=0.95, description="Confidence threshold to stop questioning")
    
    class Config:
        json_schema_extra = {
            "example": {
                "Age": 45,
                "Gender": "M",
                "Chief_Complaint": "chest pain",
                "Symptoms": "mild discomfort",
                "max_questions": 10,
                "confidence_threshold": 0.75
            }
        }


class SessionAnswerRequest(BaseModel):
    """Request to answer a question in an active session."""
    session_id: str
    feature: str
    value: Any
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "feature": "Systolic_BP",
                "value": 120
            }
        }


class QuestionResponse(BaseModel):
    """Response for active questioning."""
    session_id: str
    done: bool
    question_feature: Optional[str] = None
    question_text: Optional[str] = None
    question_type: Optional[str] = None
    current_risk_prediction: Optional[str] = None
    current_risk_probabilities: Optional[Dict[str, float]] = None
    current_department: Optional[str] = None
    confidence: Optional[float] = None
    questions_asked: int
    rule_triggered: bool = False
    rule_reasons: List[str] = Field(default_factory=list)
    nl_explanation: Optional[str] = None
    top_features: List[List[Union[str, float]]] = Field(default_factory=list)
    error: Optional[str] = None


class MonitorUpdateRequest(BaseModel):
    """Request for temporal risk monitoring update."""
    patient_id: str
    Age: float
    Gender: str
    Chief_Complaint: str
    Symptoms: str
    Systolic_BP: float
    Heart_Rate: float
    Body_Temperature_C: float
    Respiratory_Rate: float
    SpO2: float
    Onset_Duration_Value: float
    Onset_Duration_Unit: str
    Allergies: Optional[str] = None
    Current_Medications: Optional[str] = None
    Pre_Existing_Conditions: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "P00001",
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
        }


class MonitorUpdateResponse(BaseModel):
    """Response for temporal risk monitoring update."""
    risk_level: str
    risk_probabilities: Dict[str, float]
    department: str
    deterioration: Dict[str, Any]
    timeline_length: int


class EHRUploadResponse(BaseModel):
    """Response for EHR document upload."""
    source: str = "ehr_document"
    file_format: str
    extracted_data: Dict[str, Any]
    risk_level: str
    risk_source: str
    risk_probabilities: Optional[Dict[str, float]] = None
    department: str
    department_source: str
    rule_reasons: List[str]
    top_features: List[List]
    department_clinical_probs: Optional[Dict[str, float]] = None
    department_operational_scores: Optional[Dict[str, float]] = None
    department_load: Optional[Dict[str, float]] = None
    comorbidity_effects: List[ComorbidityEffectModel]
    nl_explanation: str
    processing_time_ms: Optional[float] = None


class FHIRAPIResponse(BaseModel):
    """Response for FHIR API endpoint."""
    source: str = "fhir_api"
    extracted_data: Dict[str, Any]
    risk_level: str
    risk_source: str
    risk_probabilities: Optional[Dict[str, float]] = None
    department: str
    department_source: str
    rule_reasons: List[str]
    top_features: List[List]
    department_clinical_probs: Optional[Dict[str, float]] = None
    department_operational_scores: Optional[Dict[str, float]] = None
    department_load: Optional[Dict[str, float]] = None
    comorbidity_effects: List[ComorbidityEffectModel]
    nl_explanation: str


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models and artifacts at application startup."""
    global artifacts, feature_importance_ranking
    try:
        logger.info("Loading ML models and artifacts...")
        artifacts = load_artifacts()
        logger.info("✓ Models loaded successfully")
        
        # Compute global feature importance for active questioning
        logger.info("Computing global feature importance...")
        feature_importance_ranking = compute_global_feature_importance(artifacts)
        logger.info(f"✓ Feature importance computed: {len(feature_importance_ranking)} features")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down triage prediction service...")


# ============================================================================
# API Routes
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Triage Prediction API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": artifacts is not None
    }


class RelatedDiseasesGraphNode(BaseModel):
    """Node in the related diseases graph."""
    id: str
    label: str
    type: str  # "condition" or "symptom"


class RelatedDiseasesGraphEdge(BaseModel):
    """Edge in the related diseases graph (condition -> symptom with effect)."""
    source: str
    target: str
    weight: int
    explanation: str
    preferred_department: Optional[str] = None


class RelatedDiseasesGraphResponse(BaseModel):
    """Response for the related diseases graph (nodes and edges from comorbidity rules)."""
    nodes: List[RelatedDiseasesGraphNode]
    edges: List[RelatedDiseasesGraphEdge]


@app.get("/triage/related-diseases-graph", response_model=RelatedDiseasesGraphResponse)
async def get_related_diseases_graph():
    """
    Return a graph of related diseases (conditions) and symptoms from comorbidity rules.
    Nodes are conditions (e.g. CAD, COPD) and symptoms/complaints (e.g. chest pain).
    Edges represent risk interactions with weight (extra_risk_points) and explanation.
    """
    node_ids: Dict[str, RelatedDiseasesGraphNode] = {}
    edges: List[RelatedDiseasesGraphEdge] = []

    for (comorbidity, complaint_keyword), effect in COMORBIDITY_RULES.items():
        c_id = f"cond_{comorbidity}"
        s_id = f"symptom_{complaint_keyword.replace(' ', '_')}"
        if c_id not in node_ids:
            node_ids[c_id] = RelatedDiseasesGraphNode(id=c_id, label=comorbidity, type="condition")
        if s_id not in node_ids:
            node_ids[s_id] = RelatedDiseasesGraphNode(id=s_id, label=complaint_keyword, type="symptom")
        edges.append(RelatedDiseasesGraphEdge(
            source=c_id,
            target=s_id,
            weight=effect.extra_risk_points,
            explanation=effect.explanation,
            preferred_department=effect.preferred_department,
        ))

    # Ensure all COMORBIDITY_CODES appear as nodes even if no rule targets them
    for code in COMORBIDITY_CODES:
        c_id = f"cond_{code}"
        if c_id not in node_ids:
            node_ids[c_id] = RelatedDiseasesGraphNode(id=c_id, label=code, type="condition")

    return RelatedDiseasesGraphResponse(nodes=list(node_ids.values()), edges=edges)


@app.post("/triage/predict", response_model=TriageResponse)
async def predict_triage(patient: PatientRequest):
    """
    Predict triage risk level and department assignment for a patient.
    
    This endpoint:
    1. Applies rule-based triage logic
    2. If no rules apply, uses ML ensemble models
    3. Generates natural language explanation using Groq LLM
    
    Args:
        patient: Patient data including demographics, vitals, and medical history
    
    Returns:
        Triage prediction with risk level, department, and explanation
    """
    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please check server logs."
        )
    
    try:
        # Convert Pydantic model to dict
        patient_dict = patient.dict(exclude_none=True)
        
        # Run prediction pipeline (rules + ML ensemble)
        result = predict_risk_and_department(patient_dict, artifacts)
        
        # Generate natural language explanation using Groq LLM
        nl_explanation = "Unable to generate explanation at this time."
        try:
            nl_explanation = call_llm(result["llm_prompt"])
            logger.info("✓ LLM explanation generated successfully")
        except Exception as e:
            logger.warning(f"LLM call failed: {e}. Using fallback explanation.")
            # Use a safe fallback message
            if result["rule_reasons"]:
                nl_explanation = (
                    f"Based on clinical rules: {', '.join(result['rule_reasons'])}. "
                    f"Patient assigned to {result['department']} with {result['risk_level']} risk level."
                )
            else:
                nl_explanation = (
                    f"Patient assessed using ML models. "
                    f"Assigned to {result['department']} with {result['risk_level']} risk level."
                )
        
        # Convert top_features tuples to lists for JSON serialization
        top_features_list = [[name, value] for name, value in result["top_features"]]
        
        # Convert comorbidity effects to Pydantic models
        comorbidity_effects_list = [
            ComorbidityEffectModel(**effect) 
            for effect in result.get("comorbidity_effects", [])
        ]
        
        # Build response (exclude llm_prompt from API response)
        response = TriageResponse(
            risk_level=result["risk_level"],
            risk_source=result["risk_source"],
            risk_probabilities=result["risk_probabilities"],
            department=result["department"],
            department_source=result["department_source"],
            rule_reasons=result["rule_reasons"],
            top_features=top_features_list,
            department_clinical_probs=result.get("department_clinical_probs"),
            department_operational_scores=result.get("department_operational_scores"),
            department_load=result.get("department_load"),
            comorbidity_effects=comorbidity_effects_list,
            nl_explanation=nl_explanation
        )
        
        return response
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ============================================================================
# Active Questioning Endpoints
# ============================================================================

@app.post("/triage/session/start", response_model=QuestionResponse)
async def start_session(request: SessionStartRequest):
    """
    Start a new active questioning triage session.
    
    This endpoint begins a progressive questioning flow where the system asks
    the most informative questions based on feature importance until confidence
    threshold is reached or max questions asked.
    
    This is a deterministic, greedy questioning strategy (NOT reinforcement learning):
    - Questions are selected based on global SHAP feature importance
    - Higher importance features are asked first
    - Stops when confidence threshold reached or max questions asked
    
    Args:
        request: Initial patient information (Age, Gender, Chief_Complaint, etc.)
    
    Returns:
        Session info and first question (if any)
    """
    if artifacts is None or feature_importance_ranking is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please check server logs."
        )
    
    try:
        initial_answers = request.dict(exclude={"max_questions", "confidence_threshold", "session_id"})
        
        # Filter out None values so we don't pollute known_answers with Nones (which logic treats as "answered")
        # BUT WAIT: If logic treats "in known_answers" as answered, then None means "we know it is None"?
        # Actually, if we pass None, it means we don't know it. 
        # So we should ONLY pass non-None values to known_answers if we want the system to ask about them later if they are important.
        # However, if we sent None, we are explicitly saying "I don't have this". 
        # The prompt says "It is asking for ... even though it is present". implies we ARE sending it.
        # Let's log it.
        initial_answers = {k: v for k, v in initial_answers.items() if v is not None}
        
        # Initialize session
        session = initialize_session(
            initial_answers=initial_answers,
            max_questions=request.max_questions
        )
        session["confidence_threshold"] = request.confidence_threshold
        
        # Save session
        save_session(session)
        
        # Select first question
        question_result = select_next_question(
            session=session,
            artifacts=artifacts,
            importance_ranking=feature_importance_ranking,
            confidence_threshold=request.confidence_threshold
        )
        
        # Update session
        save_session(session)
        
        # If done, generate LLM explanation
        nl_explanation = None
        if question_result.get("done"):
            # Build final patient dict and get explanation
            from active_questioning import build_patient_dict_from_session
            patient = build_patient_dict_from_session(session)
            final_result = predict_risk_and_department(patient, artifacts)
            
            try:
                nl_explanation = call_llm(final_result["llm_prompt"])
            except Exception as e:
                logger.warning(f"LLM call failed: {e}")
                nl_explanation = "Unable to generate explanation at this time."
        
        response = QuestionResponse(
            session_id=session["session_id"],
            done=question_result.get("done", False),
            question_feature=question_result.get("question_feature"),
            question_text=question_result.get("question_text"),
            question_type=question_result.get("question_type"),
            current_risk_prediction=question_result.get("current_risk_prediction"),
            current_risk_probabilities=question_result.get("current_risk_probabilities"),
            current_department=question_result.get("current_department"),
            confidence=question_result.get("confidence"),
            questions_asked=question_result.get("questions_asked", 0),
            rule_triggered=question_result.get("rule_triggered", False),
            rule_reasons=question_result.get("rule_reasons", []),
            nl_explanation=nl_explanation,
            top_features=[list(item) for item in final_result.get("top_features", [])] if nl_explanation else [],
            error=question_result.get("error")
        )
        
        return response
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Session start error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/triage/session/next", response_model=QuestionResponse)
async def answer_question(request: SessionAnswerRequest):
    """
    Answer a question and get the next question or final decision.
    
    Args:
        request: Session ID, feature name, and answer value
    
    Returns:
        Next question or final triage decision
    """
    if artifacts is None or feature_importance_ranking is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please check server logs."
        )
    
    try:
        # Get session
        session = get_session(request.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update session with answer
        session = update_session_with_answer(
            session=session,
            feature=request.feature,
            value=request.value
        )
        
        # Select next question
        question_result = select_next_question(
            session=session,
            artifacts=artifacts,
            importance_ranking=feature_importance_ranking,
            confidence_threshold=session.get("confidence_threshold", 0.75)
        )
        
        # Update session
        save_session(session)
        
        # If done, generate LLM explanation
        nl_explanation = None
        if question_result.get("done"):
            # Build final patient dict and get explanation
            from active_questioning import build_patient_dict_from_session
            patient = build_patient_dict_from_session(session)
            final_result = predict_risk_and_department(patient, artifacts)
            
            try:
                nl_explanation = call_llm(final_result["llm_prompt"])
            except Exception as e:
                logger.warning(f"LLM call failed: {e}")
                nl_explanation = "Unable to generate explanation at this time."
        
        response = QuestionResponse(
            session_id=session["session_id"],
            done=question_result.get("done", False),
            question_feature=question_result.get("question_feature"),
            question_text=question_result.get("question_text"),
            question_type=question_result.get("question_type"),
            current_risk_prediction=question_result.get("current_risk_prediction"),
            current_risk_probabilities=question_result.get("current_risk_probabilities"),
            current_department=question_result.get("current_department"),
            confidence=question_result.get("confidence"),
            questions_asked=question_result.get("questions_asked", len(session["asked_questions"])),
            rule_triggered=question_result.get("rule_triggered", False),
            rule_reasons=question_result.get("rule_reasons", []),
            nl_explanation=nl_explanation,
            top_features=[list(item) for item in final_result.get("top_features", [])] if nl_explanation else [],
            error=question_result.get("error")
        )
        
        return response
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session next error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.delete("/triage/session/{session_id}")
async def end_session(session_id: str):
    """End and delete a session."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


# ============================================================================
# Temporal Risk Monitoring Endpoints
# ============================================================================

@app.post("/triage/monitor/update", response_model=MonitorUpdateResponse)
async def update_monitoring(request: MonitorUpdateRequest):
    """
    Update temporal risk monitoring for a patient.
    
    This endpoint:
    1. Looks up or creates a risk timeline for the patient
    2. Runs prediction on current patient data
    3. Adds snapshot to timeline
    4. Computes deterioration flags
    5. Returns prediction with temporal context
    
    Args:
        request: Patient ID and full patient observation data
    
    Returns:
        Prediction result with deterioration information
    """
    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please check server logs."
        )
    
    try:
        # Get or create timeline for this patient
        timeline = get_or_create_timeline(request.patient_id)
        
        # Convert request to patient dict (excluding patient_id)
        patient_dict = request.dict(exclude={"patient_id"})
        
        # Update timeline with new prediction
        result = update_timeline_with_prediction(
            timeline=timeline,
            patient=patient_dict,
            artifacts=artifacts
        )
        
        # Save updated timeline
        save_timeline(result["timeline"])
        
        # Get prediction result
        prediction = result["prediction"]
        deterioration = result["deterioration"]
        
        # Generate LLM explanation with temporal info
        from triage_inference import build_llm_explanation_prompt
        llm_prompt = build_llm_explanation_prompt(
            patient=patient_dict,
            risk_level=prediction["risk_level"],
            risk_probabilities=prediction.get("risk_probabilities"),
            department=prediction["department"],
            rule_reasons=prediction.get("rule_reasons", []),
            top_features=prediction.get("top_features", []),
            dept_clinical_probs=prediction.get("department_clinical_probs"),
            dept_operational_scores=prediction.get("department_operational_scores"),
            dept_load=prediction.get("department_load"),
            comorbidity_effects=prediction.get("comorbidity_effects"),
            deterioration_info=deterioration
        )
        
        # Call LLM for explanation (optional - not in response model, but could be added)
        # For now, we'll just return the structured data
        
        # Build response
        response = MonitorUpdateResponse(
            risk_level=prediction["risk_level"],
            risk_probabilities=prediction.get("risk_probabilities", {}),
            department=prediction["department"],
            deterioration=deterioration,
            timeline_length=len(result["timeline"]["snapshots"])
        )
        
        return response
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Monitor update error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/triage/monitor/timeline/{patient_id}")
async def get_patient_timeline(patient_id: str):
    """
    Get full risk timeline for a patient.
    
    Args:
        patient_id: Patient identifier
    
    Returns:
        Complete risk timeline with all snapshots
    """
    timeline = get_timeline(patient_id)
    
    if timeline is None:
        raise HTTPException(
            status_code=404,
            detail=f"No timeline found for patient {patient_id}"
        )
    
    # Convert datetime objects to ISO format strings for JSON serialization
    timeline_json = {
        "patient_id": timeline["patient_id"],
        "snapshots": [
            {
                "timestamp": snap["timestamp"].isoformat(),
                "risk_level": snap["risk_level"],
                "risk_probabilities": snap["risk_probabilities"],
                "department": snap["department"]
            }
            for snap in timeline["snapshots"]
        ]
    }
    
    return timeline_json


@app.delete("/triage/monitor/timeline/{patient_id}")
async def delete_patient_timeline(patient_id: str):
    """Delete risk timeline for a patient."""
    timeline = get_timeline(patient_id)
    if timeline is None:
        raise HTTPException(
            status_code=404,
            detail=f"No timeline found for patient {patient_id}"
        )
    
    delete_risk_timeline(patient_id)
    return {"status": "deleted", "patient_id": patient_id}


# ============================================================================
# EHR Integration Endpoints
# ============================================================================

@app.post("/triage/manual", response_model=TriageResponse)
async def manual_triage(patient: PatientRequest):
    """
    Manual triage endpoint - accepts structured patient data.
    
    This endpoint accepts the same fields as the standard /triage/predict
    endpoint and provides the same functionality. Useful for direct API
    calls with structured data.
    
    Args:
        patient: Patient data in structured format
    
    Returns:
        Triage prediction with all features (SHAP, operational, comorbidity, etc.)
    """
    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please check server logs."
        )
    
    try:
        # Convert Pydantic model to dict
        patient_dict = patient.dict(exclude_none=True)
        
        # Run prediction pipeline (rules + ML ensemble)
        result = predict_risk_and_department(patient_dict, artifacts)
        
        # Generate natural language explanation using Groq LLM
        nl_explanation = "Unable to generate explanation at this time."
        try:
            nl_explanation = call_llm(result["llm_prompt"])
            logger.info("✓ LLM explanation generated successfully")
        except Exception as e:
            logger.warning(f"LLM call failed: {e}. Using fallback explanation.")
            if result["rule_reasons"]:
                nl_explanation = (
                    f"Based on clinical rules: {', '.join(result['rule_reasons'])}. "
                    f"Patient assigned to {result['department']} with {result['risk_level']} risk level."
                )
            else:
                nl_explanation = (
                    f"Patient assessed using ML models. "
                    f"Assigned to {result['department']} with {result['risk_level']} risk level."
                )
        
        # Convert top_features tuples to lists for JSON serialization
        top_features_list = [[name, value] for name, value in result["top_features"]]
        
        # Convert comorbidity effects to Pydantic models
        comorbidity_effects_list = [
            ComorbidityEffectModel(**effect) 
            for effect in result.get("comorbidity_effects", [])
        ]
        
        # Build response
        response = TriageResponse(
            risk_level=result["risk_level"],
            risk_source=result["risk_source"],
            risk_probabilities=result["risk_probabilities"],
            department=result["department"],
            department_source=result["department_source"],
            rule_reasons=result["rule_reasons"],
            top_features=top_features_list,
            department_clinical_probs=result.get("department_clinical_probs"),
            department_operational_scores=result.get("department_operational_scores"),
            department_load=result.get("department_load"),
            comorbidity_effects=comorbidity_effects_list,
            nl_explanation=nl_explanation
        )
        
        return response
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Manual triage error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/triage/ehr-upload", response_model=EHRUploadResponse)
async def ehr_upload(file: UploadFile = File(...)):
    """
    Upload EHR document and run triage prediction.
    
    Supports multiple file formats:
    - .json: FHIR Bundle or simple JSON
    - .pdf: PDF documents (text extraction)
    - .txt: Plain text documents
    - .xml: HL7 CDA documents (stub parser)
    
    Args:
        file: Uploaded document file
    
    Returns:
        Triage prediction with extracted data and all features
    """
    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please check server logs."
        )
    
    start_time = time.time()
    
    try:
        # Validate file extension
        file_extension = Path(file.filename).suffix.lower()
        allowed_extensions = [".json", ".pdf", ".txt", ".xml"]
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. "
                       f"Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Read document using EHR reader
            ehr_reader = EHRDocumentReader()
            patient_dict = ehr_reader.read_document(tmp_path)
            
            # Run prediction pipeline
            result = predict_risk_and_department(patient_dict, artifacts)
            
            # Generate LLM explanation
            nl_explanation = "Unable to generate explanation at this time."
            try:
                nl_explanation = call_llm(result["llm_prompt"])
            except Exception as e:
                logger.warning(f"LLM call failed: {e}")
                if result["rule_reasons"]:
                    nl_explanation = (
                        f"Based on clinical rules: {', '.join(result['rule_reasons'])}. "
                        f"Patient assigned to {result['department']} with {result['risk_level']} risk level."
                    )
                else:
                    nl_explanation = (
                        f"Patient assessed using ML models. "
                        f"Assigned to {result['department']} with {result['risk_level']} risk level."
                    )
            
            # Convert top_features tuples to lists
            top_features_list = [[name, value] for name, value in result["top_features"]]
            
            # Convert comorbidity effects
            comorbidity_effects_list = [
                ComorbidityEffectModel(**effect) 
                for effect in result.get("comorbidity_effects", [])
            ]
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Build response
            response = EHRUploadResponse(
                source="ehr_document",
                file_format=file_extension,
                extracted_data=patient_dict,
                risk_level=result["risk_level"],
                risk_source=result["risk_source"],
                risk_probabilities=result["risk_probabilities"],
                department=result["department"],
                department_source=result["department_source"],
                rule_reasons=result["rule_reasons"],
                top_features=top_features_list,
                department_clinical_probs=result.get("department_clinical_probs"),
                department_operational_scores=result.get("department_operational_scores"),
                department_load=result.get("department_load"),
                comorbidity_effects=comorbidity_effects_list,
                nl_explanation=nl_explanation,
                processing_time_ms=processing_time_ms
            )
            
            return response
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"EHR upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/triage/fhir-api", response_model=FHIRAPIResponse)
async def fhir_api(fhir_bundle: dict):
    """
    Accept FHIR Bundle JSON and run triage prediction.
    
    This endpoint accepts a raw FHIR R4 Bundle in the request body
    and extracts triage-relevant patient data.
    
    Args:
        fhir_bundle: FHIR R4 Bundle as JSON dict
    
    Returns:
        Triage prediction with extracted data and all features
    """
    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please check server logs."
        )
    
    try:
        # Parse FHIR Bundle
        fhir_parser = FHIRParser()
        patient_dict = fhir_parser.parse_fhir_bundle(fhir_bundle)
        
        # Run prediction pipeline
        result = predict_risk_and_department(patient_dict, artifacts)
        
        # Generate LLM explanation
        nl_explanation = "Unable to generate explanation at this time."
        try:
            nl_explanation = call_llm(result["llm_prompt"])
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            if result["rule_reasons"]:
                nl_explanation = (
                    f"Based on clinical rules: {', '.join(result['rule_reasons'])}. "
                    f"Patient assigned to {result['department']} with {result['risk_level']} risk level."
                )
            else:
                nl_explanation = (
                    f"Patient assessed using ML models. "
                    f"Assigned to {result['department']} with {result['risk_level']} risk level."
                )
        
        # Convert top_features tuples to lists
        top_features_list = [[name, value] for name, value in result["top_features"]]
        
        # Convert comorbidity effects
        comorbidity_effects_list = [
            ComorbidityEffectModel(**effect) 
            for effect in result.get("comorbidity_effects", [])
        ]
        
        # Build response
        response = FHIRAPIResponse(
            source="fhir_api",
            extracted_data=patient_dict,
            risk_level=result["risk_level"],
            risk_source=result["risk_source"],
            risk_probabilities=result["risk_probabilities"],
            department=result["department"],
            department_source=result["department_source"],
            rule_reasons=result["rule_reasons"],
            top_features=top_features_list,
            department_clinical_probs=result.get("department_clinical_probs"),
            department_operational_scores=result.get("department_operational_scores"),
            department_load=result.get("department_load"),
            comorbidity_effects=comorbidity_effects_list,
            nl_explanation=nl_explanation
        )
        
        return response
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid FHIR Bundle: {str(e)}")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Malformed FHIR Bundle: missing {str(e)}")
    except Exception as e:
        logger.error(f"FHIR API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
