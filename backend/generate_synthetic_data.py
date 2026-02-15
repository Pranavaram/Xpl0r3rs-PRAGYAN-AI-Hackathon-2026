#!/usr/bin/env python3
"""
Script to generate synthetic medical triage data using SDV (CTGAN/CopulaGAN)
from a seed dataset of 550 samples to generate 20k high-quality samples.
"""

import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer, CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath):
    """Load the seed dataset and prepare it for SDV."""
    print("Loading seed dataset...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    return df

def create_metadata(df):
    """Create SDV metadata for the dataset."""
    print("\nCreating SDV metadata...")
    metadata = SingleTableMetadata()
    
    # Detect metadata from dataframe (without PatientID first)
    df_for_detection = df.copy()
    if 'PatientID' in df_for_detection.columns:
        # Temporarily remove PatientID for detection
        patient_ids = df_for_detection['PatientID'].copy()
        df_for_detection = df_for_detection.drop(columns=['PatientID'])
    
    metadata.detect_from_dataframe(df_for_detection)
    
    # Now add PatientID as 'id' type and set as primary key
    if 'PatientID' in df.columns:
        metadata.add_column('PatientID', sdtype='id')
        metadata.set_primary_key('PatientID')
    
    # Ensure numerical columns are properly identified
    numerical_cols = ['Age', 'Systolic_BP', 'Heart_Rate', 'Body_Temperature_C', 
                     'Respiratory_Rate', 'SpO2', 'Onset_Duration_Value']
    for col in numerical_cols:
        if col in metadata.columns and col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                metadata.columns[col]['sdtype'] = 'numerical'
    
    # Ensure categorical columns are properly identified
    categorical_cols = ['Gender', 'Chief_Complaint', 'Onset_Duration_Unit', 
                       'Allergies', 'Current_Medications', 'Pre_Existing_Conditions',
                       'Risk_Level', 'Department']
    for col in categorical_cols:
        if col in metadata.columns and col in df.columns:
            metadata.columns[col]['sdtype'] = 'categorical'
    
    # Symptoms might be text, but we'll treat as categorical for now
    if 'Symptoms' in metadata.columns:
        metadata.columns['Symptoms']['sdtype'] = 'categorical'
    
    # Validate metadata
    try:
        metadata.validate()
        print("Metadata created and validated successfully")
    except Exception as e:
        print(f"Metadata validation warning: {e}")
        print("Continuing anyway...")
    
    return metadata

def train_and_generate_ctgan(df, metadata, num_samples=20000):
    """Train CTGAN model and generate synthetic data."""
    print(f"\n{'='*60}")
    print("Training CTGAN model...")
    print(f"{'='*60}")
    
    # Initialize CTGAN synthesizer with optimized parameters for quality
    synthesizer = CTGANSynthesizer(
        metadata=metadata,
        epochs=300,  # More epochs for better quality
        batch_size=500,
        verbose=True
    )
    
    # Train the model
    synthesizer.fit(df)
    print("\nModel training completed!")
    
    # Generate synthetic data
    print(f"\nGenerating {num_samples} synthetic samples...")
    synthetic_data = synthesizer.sample(num_rows=num_samples)
    
    return synthetic_data, synthesizer

def train_and_generate_copulagan(df, metadata, num_samples=20000):
    """Train CopulaGAN model and generate synthetic data."""
    print(f"\n{'='*60}")
    print("Training CopulaGAN model...")
    print(f"{'='*60}")
    
    # Initialize CopulaGAN synthesizer
    synthesizer = CopulaGANSynthesizer(
        metadata=metadata,
        epochs=300,  # More epochs for better quality
        verbose=True
    )
    
    # Train the model
    synthesizer.fit(df)
    print("\nModel training completed!")
    
    # Generate synthetic data
    print(f"\nGenerating {num_samples} synthetic samples...")
    synthetic_data = synthesizer.sample(num_rows=num_samples)
    
    return synthetic_data, synthesizer

def validate_synthetic_data(original_df, synthetic_df):
    """Validate the quality of synthetic data."""
    print(f"\n{'='*60}")
    print("Validating synthetic data quality...")
    print(f"{'='*60}")
    
    print(f"\nOriginal dataset shape: {original_df.shape}")
    print(f"Synthetic dataset shape: {synthetic_df.shape}")
    
    # Check for missing columns
    missing_cols = set(original_df.columns) - set(synthetic_df.columns)
    if missing_cols:
        print(f"WARNING: Missing columns in synthetic data: {missing_cols}")
    else:
        print("✓ All columns present")
    
    # Check data types
    print("\nData type comparison:")
    for col in original_df.columns:
        if col in synthetic_df.columns:
            orig_type = original_df[col].dtype
            synth_type = synthetic_df[col].dtype
            match = "✓" if orig_type == synth_type else "⚠"
            print(f"  {match} {col}: {orig_type} -> {synth_type}")
    
    # Check value ranges for numerical columns
    numerical_cols = ['Age', 'Systolic_BP', 'Heart_Rate', 'Body_Temperature_C', 
                     'Respiratory_Rate', 'SpO2', 'Onset_Duration_Value']
    print("\nNumerical column ranges:")
    for col in numerical_cols:
        if col in original_df.columns and col in synthetic_df.columns:
            orig_min, orig_max = original_df[col].min(), original_df[col].max()
            synth_min, synth_max = synthetic_df[col].min(), synthetic_df[col].max()
            print(f"  {col}:")
            print(f"    Original: [{orig_min:.2f}, {orig_max:.2f}]")
            print(f"    Synthetic: [{synth_min:.2f}, {synth_max:.2f}]")
    
    # Check categorical distributions
    categorical_cols = ['Gender', 'Risk_Level', 'Department']
    print("\nCategorical column distributions:")
    for col in categorical_cols:
        if col in original_df.columns and col in synthetic_df.columns:
            print(f"\n  {col}:")
            orig_counts = original_df[col].value_counts(normalize=True).head(5)
            synth_counts = synthetic_df[col].value_counts(normalize=True).head(5)
            print(f"    Original top 5:")
            for val, pct in orig_counts.items():
                print(f"      {val}: {pct*100:.1f}%")
            print(f"    Synthetic top 5:")
            for val, pct in synth_counts.items():
                print(f"      {val}: {pct*100:.1f}%")

def main():
    """Main function to generate synthetic data."""
    input_file = "triage_seed_dataset_with_edges.csv"
    output_file = "triage_synthetic_dataset_20k.csv"
    
    # Load and prepare data
    df = load_and_prepare_data(input_file)
    
    # Create metadata
    metadata = create_metadata(df)
    
    # Try CTGAN first (often better for mixed data types)
    print("\n" + "="*60)
    print("ATTEMPTING CTGAN SYNTHESIS")
    print("="*60)
    try:
        synthetic_data, model = train_and_generate_ctgan(df, metadata, num_samples=20000)
        model_name = "CTGAN"
    except Exception as e:
        print(f"\nCTGAN failed with error: {e}")
        print("\nTrying CopulaGAN instead...")
        try:
            synthetic_data, model = train_and_generate_copulagan(df, metadata, num_samples=20000)
            model_name = "CopulaGAN"
        except Exception as e2:
            print(f"\nCopulaGAN also failed with error: {e2}")
            raise
    
    # Validate the synthetic data
    validate_synthetic_data(df, synthetic_data)
    
    # Save the synthetic data
    print(f"\n{'='*60}")
    print(f"Saving synthetic data to {output_file}...")
    print(f"{'='*60}")
    synthetic_data.to_csv(output_file, index=False)
    print(f"\n✓ Successfully generated {len(synthetic_data)} synthetic samples using {model_name}")
    print(f"✓ Saved to: {output_file}")
    
    # Generate new PatientIDs for synthetic data (optional, but good practice)
    print("\nGenerating new unique PatientIDs...")
    synthetic_data['PatientID'] = [f"syn_{i:08x}" for i in range(len(synthetic_data))]
    synthetic_data.to_csv(output_file, index=False)
    print("✓ Updated PatientIDs in synthetic dataset")
    
    return synthetic_data

if __name__ == "__main__":
    synthetic_data = main()
