"""
EHR Document Reader
Universal reader for EHR documents in multiple formats.

Supports: .json (FHIR), .pdf, .txt, .xml (HL7 CDA stub)

Note: PDF/text parsing uses simple regex-based extraction.
In production, this would be replaced with NLP/LLM-based extraction
for better accuracy and handling of unstructured text.
"""

import os
import re
import json
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional
from pathlib import Path

from .fhir_parser import FHIRParser

# Try to import PyPDF2, but make it optional
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# ============================================================================
# Document Reader Class
# ============================================================================

class EHRDocumentReader:
    """
    Universal reader for EHR documents.
    Supports: .json (FHIR), .pdf, .txt, .xml (HL7 CDA stub)
    
    Returns a triage-ready patient dict compatible with predict_risk_and_department.
    """
    
    def __init__(self):
        """Initialize the document reader."""
        self.fhir_parser = FHIRParser()
    
    def read_document(self, file_path: str) -> dict:
        """
        Detect file extension and dispatch to the appropriate reader.
        
        Args:
            file_path: Path to the document file
        
        Returns:
            Triage-ready patient dict
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension == ".json":
            return self._read_json(file_path)
        elif extension == ".pdf":
            return self._read_pdf(file_path)
        elif extension == ".txt":
            return self._read_txt(file_path)
        elif extension == ".xml":
            return self._read_xml(file_path)
        else:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported formats: .json, .pdf, .txt, .xml"
            )
    
    def _read_json(self, file_path: str) -> dict:
        """
        Read JSON file - detect if it's FHIR and parse accordingly.
        
        Args:
            file_path: Path to JSON file
        
        Returns:
            Triage-ready patient dict
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if it's a FHIR Bundle or resource
        resource_type = data.get("resourceType")
        if resource_type in ["Bundle", "Patient", "Observation", "Condition", 
                            "MedicationStatement", "AllergyIntolerance"]:
            # Use FHIR parser
            return self.fhir_parser.parse_fhir_bundle(data)
        else:
            # Try to parse as a simple JSON with triage fields
            # This allows for direct JSON input matching our schema
            return self._parse_simple_json(data)
    
    def _parse_simple_json(self, data: dict) -> dict:
        """
        Parse a simple JSON dict that may already be in triage format.
        Fills in missing fields with defaults.
        """
        # Default values
        result = {
            "PatientID": data.get("PatientID"),
            "Age": data.get("Age"),
            "Gender": data.get("Gender", "Unknown"),
            "Chief_Complaint": data.get("Chief_Complaint"),
            "Symptoms": data.get("Symptoms"),
            "Systolic_BP": data.get("Systolic_BP"),
            "Heart_Rate": data.get("Heart_Rate"),
            "Body_Temperature_C": data.get("Body_Temperature_C"),
            "Respiratory_Rate": data.get("Respiratory_Rate"),
            "SpO2": data.get("SpO2"),
            "Onset_Duration_Value": data.get("Onset_Duration_Value", 1.0),
            "Onset_Duration_Unit": data.get("Onset_Duration_Unit", "hours"),
            "Allergies": data.get("Allergies", "None"),
            "Current_Medications": data.get("Current_Medications", "None"),
            "Pre_Existing_Conditions": data.get("Pre_Existing_Conditions", "None")
        }
        
        return result
    
    def _read_pdf(self, file_path: str) -> dict:
        """
        Read PDF file and extract text using simple regex parsing.
        
        Note: This is a simplified parser. In production, use NLP/LLM
        for better extraction from unstructured PDF documents.
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            Triage-ready patient dict
        """
        if not PDF_AVAILABLE:
            raise ImportError(
                "PyPDF2 is required for PDF parsing. "
                "Install it with: pip install PyPDF2"
            )
        
        # Extract text from PDF
        text = ""
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise ValueError(f"Failed to read PDF: {e}")
        
        # Parse text using regex
        return self._parse_text(text)
    
    def _read_txt(self, file_path: str) -> dict:
        """
        Read text file and extract patient data using regex parsing.
        
        Note: This is a simplified parser. In production, use NLP/LLM
        for better extraction from unstructured text documents.
        
        Args:
            file_path: Path to text file
        
        Returns:
            Triage-ready patient dict
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self._parse_text(text)
    
    def _parse_text(self, text: str) -> dict:
        """
        Parse unstructured text to extract triage-relevant fields.
        
        Uses simple regex patterns. In production, replace with NLP/LLM.
        
        Args:
            text: Raw text content
        
        Returns:
            Triage-ready patient dict
        """
        # Initialize with defaults
        result = {
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
            "Onset_Duration_Value": 1.0,
            "Onset_Duration_Unit": "hours",
            "Allergies": "None",
            "Current_Medications": "None",
            "Pre_Existing_Conditions": "None"
        }
        
        text_lower = text.lower()
        
        # Extract Age
        age_patterns = [
            r'age[:\s]+(\d+)',
            r'(\d+)\s*years?\s*old',
            r'(\d+)\s*yo',
            r'age[:\s]+(\d+)'
        ]
        for pattern in age_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                try:
                    result["Age"] = float(match.group(1))
                    break
                except (ValueError, IndexError):
                    pass
        
        # Extract Gender
        if re.search(r'\b(male|m\b|man)', text_lower):
            result["Gender"] = "M"
        elif re.search(r'\b(female|f\b|woman)', text_lower):
            result["Gender"] = "F"
        elif re.search(r'\b(transgender|trans)', text_lower):
            result["Gender"] = "T"
        
        # Extract Systolic BP
        bp_patterns = [
            r'bp[:\s]+(\d+)[/\s]+\d+',  # BP: 120/80
            r'systolic[:\s]+(\d+)',
            r'(\d+)[/\s]+\d+\s*mmhg',  # 120/80 mmHg
            r'blood\s*pressure[:\s]+(\d+)'
        ]
        for pattern in bp_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                try:
                    result["Systolic_BP"] = float(match.group(1))
                    break
                except (ValueError, IndexError):
                    pass
        
        # Extract Heart Rate
        hr_patterns = [
            r'heart\s*rate[:\s]+(\d+)',
            r'hr[:\s]+(\d+)',
            r'pulse[:\s]+(\d+)',
            r'(\d+)\s*bpm'
        ]
        for pattern in hr_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                try:
                    result["Heart_Rate"] = float(match.group(1))
                    break
                except (ValueError, IndexError):
                    pass
        
        # Extract Temperature
        temp_patterns = [
            r'temperature[:\s]+(\d+\.?\d*)\s*[cf]',
            r'temp[:\s]+(\d+\.?\d*)\s*[cf]',
            r'(\d+\.?\d*)\s*°[cf]',
            r'(\d+\.?\d*)\s*degrees?\s*[cf]'
        ]
        for pattern in temp_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                try:
                    temp = float(match.group(1))
                    # Check if Fahrenheit (convert to Celsius)
                    if 'f' in match.group(0).lower():
                        temp = (temp - 32) * 5 / 9
                    result["Body_Temperature_C"] = temp
                    break
                except (ValueError, IndexError):
                    pass
        
        # Extract Respiratory Rate
        rr_patterns = [
            r'respiratory\s*rate[:\s]+(\d+)',
            r'rr[:\s]+(\d+)',
            r'breathing[:\s]+(\d+)',
            r'(\d+)\s*breaths?\s*per\s*min'
        ]
        for pattern in rr_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                try:
                    result["Respiratory_Rate"] = float(match.group(1))
                    break
                except (ValueError, IndexError):
                    pass
        
        # Extract SpO2
        spo2_patterns = [
            r'spo2[:\s]+(\d+)',
            r'oxygen\s*saturation[:\s]+(\d+)',
            r'o2\s*sat[:\s]+(\d+)',
            r'(\d+)%\s*o2'
        ]
        for pattern in spo2_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                try:
                    result["SpO2"] = float(match.group(1))
                    break
                except (ValueError, IndexError):
                    pass
        
        # Extract Chief Complaint (look for common patterns)
        complaint_patterns = [
            r'chief\s*complaint[:\s]+(.+?)(?:\n|$)',
            r'presenting\s*complaint[:\s]+(.+?)(?:\n|$)',
            r'cc[:\s]+(.+?)(?:\n|$)',
            r'reason\s*for\s*visit[:\s]+(.+?)(?:\n|$)'
        ]
        for pattern in complaint_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
            if match:
                result["Chief_Complaint"] = match.group(1).strip()
                break
        
        # Extract Symptoms (look for symptoms section)
        symptoms_patterns = [
            r'symptoms[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)',
            r'symptom[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)'
        ]
        for pattern in symptoms_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                result["Symptoms"] = match.group(1).strip()[:500]  # Limit length
                break
        
        # Extract Allergies
        allergy_patterns = [
            r'allerg(?:y|ies)[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)',
            r'no\s*known\s*allerg(?:y|ies)',
            r'nka'
        ]
        for pattern in allergy_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
            if match:
                if 'no known' in match.group(0).lower() or 'nka' in match.group(0).lower():
                    result["Allergies"] = "None"
                else:
                    result["Allergies"] = match.group(1).strip() if match.lastindex else "None"
                break
        
        # Extract Medications
        med_patterns = [
            r'medication(?:s)?[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)',
            r'current\s*medication(?:s)?[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)',
            r'meds[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)'
        ]
        for pattern in med_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                result["Current_Medications"] = match.group(1).strip()[:500]  # Limit length
                break
        
        # Extract Pre-existing Conditions
        condition_patterns = [
            r'past\s*medical\s*history[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)',
            r'pmh[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)',
            r'history[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)',
            r'chronic\s*condition(?:s)?[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)'
        ]
        for pattern in condition_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                result["Pre_Existing_Conditions"] = match.group(1).strip()[:500]  # Limit length
                break
        
        return result
    
    def _read_xml(self, file_path: str) -> dict:
        """
        Read XML file (HL7 CDA stub parser).
        
        Note: This is a minimal HL7 CDA parser. In production, use
        a proper HL7 library for comprehensive parsing.
        
        Args:
            file_path: Path to XML file
        
        Returns:
            Triage-ready patient dict
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except Exception as e:
            raise ValueError(f"Failed to parse XML: {e}")
        
        # Initialize with defaults
        result = {
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
            "Onset_Duration_Value": 1.0,
            "Onset_Duration_Unit": "hours",
            "Allergies": "None",
            "Current_Medications": "None",
            "Pre_Existing_Conditions": "None"
        }
        
        # Try to extract patient ID (common HL7 CDA patterns)
        # This is a simplified extraction - real HL7 parsing would be more complex
        ns = {'hl7': 'urn:hl7-org:v3'}  # Common HL7 namespace
        
        # Look for patient ID
        id_elements = root.findall('.//{urn:hl7-org:v3}id') or root.findall('.//id')
        if id_elements:
            patient_id = id_elements[0].get('extension') or id_elements[0].text
            if patient_id:
                result["PatientID"] = patient_id
        
        # Look for gender
        gender_elements = root.findall('.//{urn:hl7-org:v3}administrativeGenderCode') or root.findall('.//administrativeGenderCode')
        if gender_elements:
            gender_code = gender_elements[0].get('code', '').lower()
            if gender_code == 'M':
                result["Gender"] = "M"
            elif gender_code == 'F':
                result["Gender"] = "F"
        
        # For HL7 CDA, we mark it as a stub
        # In production, implement full HL7 CDA parsing
        result["_source"] = "HL7 CDA (stub parser)"
        
        return result
