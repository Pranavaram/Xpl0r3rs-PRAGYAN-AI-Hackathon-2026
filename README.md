Smart Triage
Status Backend Frontend ML

An advanced clinical decision support system that uses Hybrid AI (Rule-Based + ML Ensemble) to triage patients in real-time. It features Explainable AI (XAI) to provide transparent reasoning for every prediction, helping clinicians make faster, safer decisions.

🚀 Key Features
Hybrid AI Engine: Combines a deterministic Safety Net (for immediate life threats) with a probabilistic ML Ensemble (XGBoost, LightGBM, CatBoost) for robust risk prediction.
Explainable AI (XAI): Uses SHAP (SHapley Additive exPlanations) to visualize exactly why a patient was flagged as High Risk (e.g., "Age > 65 + Chest Pain").
Dynamic Questioning: The system acts like a doctor, asking relevant follow-up questions based on initial symptoms (e.g., "radiating pain?" for chest complaints).
Multi-Modal Input: Supports Voice Dictation, Text, and PDF/EHR Uploads.
Context-Aware NLP: Uses TF-IDF vectorization to understand nuanced terms like "severe" vs. "mild" pain.
Relational Disease Mapping: Interactive graph visualization of correlated diseases.
Multilingual Support: Full UI localization for diverse patient populations.

🛠️ Tech Stack

Frontend:

React + Vite (Fast, modern UI)
TailwindCSS (Styling)
Framer Motion (Animations)
Chart.js (Data Visualization)
Backend
Python + FastAPI (High-performance API)
Pydantic (Data Validation)
Scikit-Learn / XGBoost / LightGBM / CatBoost (Machine Learning)
SHAP (Explainability)
Uvicorn (ASGI Server)


📦 Installation:

Prerequisites
Node.js (v16+)
Python (v3.9+)
1. Clone the Repository
bash
git clone https://github.com/StartUpUser/intelligent-triage-system.git
cd intelligent-triage-system
2. Backend Setup
bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
Server runs on http://localhost:8000

3. Frontend Setup
bash
cd ../frontend
npm install
npm run dev
App runs on http://localhost:5173

🧠 Model Performance
The current model (v2.0) has been optimized for High Sensitivity to ensure patient safety.

High Risk Recall: 97% (catches almost all critical cases).
Approach: Uses class weights and oversampling to handle the rarity of medical emergencies in training data.
