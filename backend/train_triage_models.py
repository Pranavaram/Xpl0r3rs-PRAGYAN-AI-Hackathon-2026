"""
Triage Dataset ML Training Script (IMPROVED)
--------------------------------------------
Updates:
1. Includes 'Symptoms' in training (previously ignored).
2. Uses TF-IDF for text features (Chief Complaint, Symptoms) instead of OneHot.
3. Adds Oversampling for 'High' risk class to improve Recall.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, recall_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

CSV_FILE = "triage_synthetic_dataset_20k.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature columns
NUMERIC_FEATURES = [
    'Age', 'Systolic_BP', 'Heart_Rate', 'Body_Temperature_C',
    'Respiratory_Rate', 'SpO2', 'Onset_Duration_Value'
]

CATEGORICAL_FEATURES = [
    'Gender', 'Onset_Duration_Unit'
]

# Text features for TF-IDF
TEXT_FEATURES = ['Chief_Complaint', 'Symptoms']

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES

# Target columns
TARGET_RISK = 'Risk_Level'
TARGET_DEPT = 'Department'

# Model save paths
MODEL_DIR = "models"
PREPROCESSOR_PATH = f"{MODEL_DIR}/preprocessor.joblib"
RISK_XGB_PATH = f"{MODEL_DIR}/risk_xgb.joblib"
RISK_LGBM_PATH = f"{MODEL_DIR}/risk_lgbm.joblib"
RISK_CAT_PATH = f"{MODEL_DIR}/risk_cat.joblib"
DEPT_XGB_PATH = f"{MODEL_DIR}/dept_xgb.joblib"
RISK_LABEL_ENCODER_PATH = f"{MODEL_DIR}/risk_label_encoder.joblib"
DEPT_LABEL_ENCODER_PATH = f"{MODEL_DIR}/dept_label_encoder.joblib"


# ============================================================================
# Data Loading and Cleaning
# ============================================================================

def load_and_clean_data(filepath):
    """Load CSV and perform cleaning operations."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Replace empty strings and whitespace with NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    # Handle string columns
    string_cols = ['Allergies', 'Current_Medications', 'Pre_Existing_Conditions', 'Symptoms']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    
    # Convert numeric columns
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Normalize Onset_Duration_Unit
    if 'Onset_Duration_Unit' in df.columns:
        unit_mapping = {
            'hour': 'hours', 'day': 'days', 'month': 'months', 'year': 'years',
            'Hour': 'hours', 'Day': 'days', 'Month': 'months', 'Year': 'years',
            'Hours': 'hours', 'Days': 'days', 'Months': 'months', 'Years': 'years',
        }
        df['Onset_Duration_Unit'] = df['Onset_Duration_Unit'].replace(unit_mapping)
        valid_units = ['hours', 'days', 'months', 'years']
        df['Onset_Duration_Unit'] = df['Onset_Duration_Unit'].apply(
            lambda x: x if x in valid_units else 'hours'
        )
    
    # Drop rows with NaN in numeric features
    initial_rows = len(df)
    df = df.dropna(subset=NUMERIC_FEATURES)
    print(f"Dropped {initial_rows - len(df)} rows due to missing numeric values")
    
    # Ensure targets are clean
    if TARGET_RISK in df.columns:
        df[TARGET_RISK] = df[TARGET_RISK].astype(str).str.strip()
    if TARGET_DEPT in df.columns:
        df[TARGET_DEPT] = df[TARGET_DEPT].astype(str).str.strip()
        
    return df

def balance_dataset(df):
    """Oversample High Risk cases to improve recall."""
    print("\nBalancing dataset...")
    
    # Separate classes
    high_risk = df[df[TARGET_RISK] == 'High']
    other_risk = df[df[TARGET_RISK] != 'High']
    
    n_high = len(high_risk)
    n_other = len(other_risk)
    print(f"  Original counts: High={n_high}, Other={n_other} (Ratio: {n_high/len(df):.1%})")
    
    # Oversample High Risk (Target ~20% of dataset or 5x multiplier)
    # Let's multiply High Risk cases by 5
    mapping = {
        'High': 5,
        'Medium': 1,
        'Low': 1
    }
    
    dfs = []
    for risk, multiplier in mapping.items():
        subset = df[df[TARGET_RISK] == risk]
        if multiplier > 1:
            subset = pd.concat([subset] * multiplier, ignore_index=True)
        dfs.append(subset)
        
    df_balanced = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)
    
    new_counts = df_balanced[TARGET_RISK].value_counts()
    print(f"  New counts: \n{new_counts}")
    
    return df_balanced

# ============================================================================
# Preprocessing Pipeline
# ============================================================================

def create_preprocessor():
    """Create preprocessing pipeline with TF-IDF for text."""
    print("\nCreating preprocessing pipeline...")
    
    # 1. Numeric: Passthrough
    numeric_transformer = 'passthrough'
    
    # 2. Categorical: One-Hot
    categorical_transformer = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False,
        drop='first'
    )
    
    # 3. Text: TF-IDF
    # We create separate vectorizers for Complaint and Symptoms to keep semantic meaning distinct
    tfidf_cc = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2)  # Capture "chest pain" as a token
    )
    
    tfidf_sym = TfidfVectorizer(
        max_features=150,  # Symptoms are richer, allow more features
        stop_words='english',
        ngram_range=(1, 1)
    )
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
            ('txt_cc', tfidf_cc, 'Chief_Complaint'),
            ('txt_sym', tfidf_sym, 'Symptoms')
        ],
        remainder='drop'
    )
    
    print("  ✓ Preprocessor created (Numeric + OneHot + TF-IDF)")
    return preprocessor


# ============================================================================
# Model Training
# ============================================================================

def train_risk_models(X_train, y_train, X_test, y_test, preprocessor):
    """Train ensemble models for Risk_Level prediction."""
    print("\nTraining Risk Models...")
    
    # Encode label
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    classes = le.classes_
    
    # Compute Weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_enc), y=y_train_enc)
    weight_dict = dict(zip(np.unique(y_train_enc), class_weights))
    sample_weights = np.array([weight_dict[y] for y in y_train_enc])
    
    # Transform Data
    print("  Transforming data...")
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    pipelines = {}
    
    # XGBoost
    print("  Training XGBoost...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
        eval_metric='mlogloss', use_label_encoder=False
    )
    xgb_clf.fit(X_train_trans, y_train_enc, sample_weight=sample_weights)
    pipelines['xgb'] = Pipeline([('preprocessor', preprocessor), ('classifier', xgb_clf)])
    
    # LightGBM
    print("  Training LightGBM...")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        random_state=RANDOM_STATE, verbose=-1, class_weight='balanced'
    )
    lgb_clf.fit(X_train_trans, y_train_enc)
    pipelines['lgbm'] = Pipeline([('preprocessor', preprocessor), ('classifier', lgb_clf)])
    
    # CatBoost
    print("  Training CatBoost...")
    cb_clf = cb.CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        random_state=RANDOM_STATE, verbose=0, auto_class_weights='Balanced'
    )
    cb_clf.fit(X_train_trans, y_train_enc)
    pipelines['cat'] = Pipeline([('preprocessor', preprocessor), ('classifier', cb_clf)])
    
    # Evaluation
    print("\nRisk Model Evaluation (Ensemble):")
    
    # Soft Voting
    p1 = pipelines['xgb']['classifier'].predict_proba(X_test_trans)
    p2 = pipelines['lgbm']['classifier'].predict_proba(X_test_trans)
    p3 = pipelines['cat']['classifier'].predict_proba(X_test_trans)
    avg_prob = (p1 + p2 + p3) / 3
    y_pred_idx = np.argmax(avg_prob, axis=1)
    y_pred = le.inverse_transform(y_pred_idx)
    
    print(classification_report(y_test, y_pred))
    
    return pipelines, classes, le

def train_dept_model(X_train, y_train, X_test, y_test, preprocessor):
    """Train Department Model."""
    print("\nTraining Department Model...")
    
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Transform
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    # XGBoost
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
        use_label_encoder=False
    )
    # Compute sample weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_enc), y=y_train_enc)
    w_dict = dict(zip(np.unique(y_train_enc), class_weights))
    s_weights = np.array([w_dict[y] for y in y_train_enc])
    
    clf.fit(X_train_trans, y_train_enc, sample_weight=s_weights)
    
    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
    
    # Eval
    y_pred = le.inverse_transform(clf.predict(X_test_trans))
    print("\nDepartment Evaluation:")
    print(classification_report(y_test, y_pred))
    
    return pipeline, le

def save_models(preprocessor, risk_models, dept_model, risk_le, dept_le):
    print(f"\nSaving models to {MODEL_DIR}...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(risk_le, RISK_LABEL_ENCODER_PATH)
    joblib.dump(dept_le, DEPT_LABEL_ENCODER_PATH)
    
    joblib.dump(risk_models['xgb'], RISK_XGB_PATH)
    joblib.dump(risk_models['lgbm'], RISK_LGBM_PATH)
    joblib.dump(risk_models['cat'], RISK_CAT_PATH)
    joblib.dump(dept_model, DEPT_XGB_PATH)
    print("✓ Models saved.")

# ============================================================================
# Main
# ============================================================================

def main():
    print("Starting Improved Triage Training...")
    
    # 1. Load
    df = load_and_clean_data(CSV_FILE)
    
    # 2. Balance (Oversample High Risk)
    df_balanced = balance_dataset(df)
    
    # 3. Features
    X = df_balanced[ALL_FEATURES]
    y_risk = df_balanced[TARGET_RISK]
    y_dept = df_balanced[TARGET_DEPT]
    
    # 4. Split
    X_train, X_test, y_risk_train, y_risk_test = train_test_split(
        X, y_risk, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_risk
    )
    y_dept_train = y_dept.loc[X_train.index]
    y_dept_test = y_dept.loc[X_test.index]
    
    # 5. Preprocessor
    preprocessor = create_preprocessor()
    preprocessor.fit(X_train)
    
    # 6. Train
    risk_models, risk_classes, risk_le = train_risk_models(
        X_train, y_risk_train, X_test, y_risk_test, preprocessor
    )
    
    dept_model, dept_le = train_dept_model(
        X_train, y_dept_train, X_test, y_dept_test, preprocessor
    )
    
    # 7. Save
    save_models(preprocessor, risk_models, dept_model, risk_le, dept_le)
    
    print("\nDone! Copy the 'models' folder to your application.")

if __name__ == "__main__":
    main()
