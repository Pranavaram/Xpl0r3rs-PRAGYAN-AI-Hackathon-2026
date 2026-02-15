"""
FHIR R4 Parser for Triage System
Parses FHIR R4 Bundles and extracts triage-ready patient data.

Note: This parser focuses on core triage features only.
In production, we would extend to more FHIR resources and handle
edge cases more comprehensively.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, date
import json


# ============================================================================
# LOINC Code Mappings
# ============================================================================

# LOINC codes for vital signs
LOINC_CODES = {
    "8480-6": "Systolic_BP",  # Systolic blood pressure
    "8462-4": "Diastolic_BP",  # Diastolic blood pressure (not used in triage, but available)
    "8867-4": "Heart_Rate",  # Heart rate
    "9279-1": "Respiratory_Rate",  # Respiratory rate
    "8310-5": "Body_Temperature_C",  # Body temperature
    "8331-1": "Body_Temperature_C",  # Alternative temperature code
    "2708-6": "SpO2",  # Oxygen saturation in Arterial blood by Pulse oximetry
    "20564-1": "SpO2",  # Alternative SpO2 code
}

# Gender mapping from FHIR to triage format
GENDER_MAP = {
    "male": "M",
    "female": "F",
    "other": "T",
    "unknown": "Unknown"
}


# ============================================================================
# FHIR Parser Class
# ============================================================================

class FHIRParser:
    """
    Parser for FHIR R4 Bundles.
    Extracts triage-relevant patient data from FHIR resources.
    """
    
    def __init__(self):
        """Initialize the FHIR parser."""
        pass
    
    def parse_fhir_bundle(self, fhir_json: dict) -> dict:
        """
        Parse a FHIR R4 Bundle into a triage-ready dict.
        
        Args:
            fhir_json: FHIR Bundle as a dict (parsed JSON)
        
        Returns:
            Triage-ready patient dict compatible with predict_risk_and_department
        """
        # Initialize result dict with defaults
        patient_dict = {
            "PatientID": None,
            "Age": None,
            "Gender": "Unknown",
            "Chief_Complaint": None,
            "Symptoms": None,
            "Systolic_BP": None,
            "Heart_Rate": None,
            "Body_Temperature_C": None,
            "Respiratory_Rate": None,
            "SpO2": None,
            "Onset_Duration_Value": 1.0,  # Default
            "Onset_Duration_Unit": "hours",  # Default
            "Allergies": "None",
            "Current_Medications": "None",
            "Pre_Existing_Conditions": "None"
        }
        
        # Extract Bundle resources
        bundle = fhir_json
        if bundle.get("resourceType") != "Bundle":
            # If it's a single resource, wrap it in a minimal Bundle
            if bundle.get("resourceType") in ["Patient", "Observation", "Condition", "MedicationStatement", "AllergyIntolerance"]:
                bundle = {
                    "resourceType": "Bundle",
                    "type": "collection",
                    "entry": [{"resource": bundle}]
                }
            else:
                raise ValueError(f"Expected FHIR Bundle or single resource, got {bundle.get('resourceType')}")
        
        entries = bundle.get("entry", [])
        
        # Separate resources by type
        patient_resources = []
        observations = []
        conditions = []
        medications = []
        allergies = []
        
        for entry in entries:
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")
            
            if resource_type == "Patient":
                patient_resources.append(resource)
            elif resource_type == "Observation":
                observations.append(resource)
            elif resource_type == "Condition":
                conditions.append(resource)
            elif resource_type == "MedicationStatement":
                medications.append(resource)
            elif resource_type == "AllergyIntolerance":
                allergies.append(resource)
        
        # Parse Patient resource
        if patient_resources:
            self._parse_patient(patient_resources[0], patient_dict)
        
        # Parse Observations (vitals)
        self._parse_observations(observations, patient_dict)
        
        # Parse Conditions (pre-existing conditions)
        self._parse_conditions(conditions, patient_dict)
        
        # Parse Medications
        self._parse_medications(medications, patient_dict)
        
        # Parse Allergies
        self._parse_allergies(allergies, patient_dict)
        
        return patient_dict
    
    def _parse_patient(self, patient: dict, patient_dict: dict) -> None:
        """Extract data from Patient resource."""
        # Patient ID
        patient_id = patient.get("id")
        if patient_id:
            patient_dict["PatientID"] = patient_id
        
        # Age from birthDate
        birth_date = patient.get("birthDate")
        if birth_date:
            try:
                # Parse date string (FHIR format: YYYY-MM-DD)
                birth = datetime.strptime(birth_date, "%Y-%m-%d").date()
                today = date.today()
                age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
                patient_dict["Age"] = float(age)
            except (ValueError, TypeError):
                pass
        
        # Gender
        gender = patient.get("gender", "").lower()
        patient_dict["Gender"] = GENDER_MAP.get(gender, "Unknown")
    
    def _parse_observations(self, observations: List[dict], patient_dict: dict) -> None:
        """Extract vital signs from Observation resources."""
        for obs in observations:
            code = obs.get("code", {})
            coding = code.get("coding", [])
            
            # Find LOINC code
            loinc_code = None
            for c in coding:
                if c.get("system") == "http://loinc.org":
                    loinc_code = c.get("code")
                    break
            
            if not loinc_code or loinc_code not in LOINC_CODES:
                continue
            
            # Get value
            value_quantity = obs.get("valueQuantity")
            if not value_quantity:
                continue
            
            value = value_quantity.get("value")
            unit = value_quantity.get("unit", "").lower()
            
            if value is None:
                continue
            
            # Map to triage field
            field_name = LOINC_CODES[loinc_code]
            
            # Handle temperature conversion (Fahrenheit to Celsius)
            if field_name == "Body_Temperature_C":
                if "f" in unit or "°f" in unit:
                    # Convert Fahrenheit to Celsius
                    value = (float(value) - 32) * 5 / 9
                patient_dict[field_name] = float(value)
            else:
                patient_dict[field_name] = float(value)
    
    def _parse_conditions(self, conditions: List[dict], patient_dict: dict) -> None:
        """Extract pre-existing conditions from Condition resources."""
        condition_names = []
        
        for condition in conditions:
            # Check if condition is active/chronic
            clinical_status = condition.get("clinicalStatus", {})
            status_code = clinical_status.get("coding", [{}])[0].get("code", "")
            
            # Include active, recurrence, relapse conditions
            if status_code in ["active", "recurrence", "relapse"]:
                code = condition.get("code", {})
                coding = code.get("coding", [])
                
                # Try to get display name
                for c in coding:
                    display = c.get("display")
                    if display:
                        condition_names.append(display)
                        break
                
                # Fallback to text
                if not condition_names or condition_names[-1] != display:
                    text = code.get("text")
                    if text:
                        condition_names.append(text)
        
        if condition_names:
            patient_dict["Pre_Existing_Conditions"] = ", ".join(condition_names)
    
    def _parse_medications(self, medications: List[dict], patient_dict: dict) -> None:
        """Extract current medications from MedicationStatement resources."""
        medication_names = []
        
        for med in medications:
            # Check if medication is active
            status = med.get("status", "").lower()
            if status in ["active", "intended", "on-hold"]:
                medication = med.get("medicationCodeableConcept", {})
                coding = medication.get("coding", [])
                
                # Try to get display name
                for c in coding:
                    display = c.get("display")
                    if display:
                        medication_names.append(display)
                        break
                
                # Fallback to text
                if not medication_names or medication_names[-1] != display:
                    text = medication.get("text")
                    if text:
                        medication_names.append(text)
        
        if medication_names:
            patient_dict["Current_Medications"] = ", ".join(medication_names)
    
    def _parse_allergies(self, allergies: List[dict], patient_dict: dict) -> None:
        """Extract allergies from AllergyIntolerance resources."""
        allergy_names = []
        
        for allergy in allergies:
            # Check if allergy is active
            clinical_status = allergy.get("clinicalStatus", {})
            status_code = clinical_status.get("coding", [{}])[0].get("code", "")
            
            if status_code == "active":
                code = allergy.get("code", {})
                coding = code.get("coding", [])
                
                # Try to get display name
                for c in coding:
                    display = c.get("display")
                    if display:
                        allergy_names.append(display)
                        break
                
                # Fallback to text
                if not allergy_names or allergy_names[-1] != display:
                    text = code.get("text")
                    if text:
                        allergy_names.append(text)
        
        if allergy_names:
            patient_dict["Allergies"] = ", ".join(allergy_names)
    
    def parse_fhir_file(self, file_path: str) -> dict:
        """
        Load FHIR JSON from file and parse it.
        
        Args:
            file_path: Path to FHIR JSON file
        
        Returns:
            Triage-ready patient dict
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            fhir_json = json.load(f)
        
        return self.parse_fhir_bundle(fhir_json)
