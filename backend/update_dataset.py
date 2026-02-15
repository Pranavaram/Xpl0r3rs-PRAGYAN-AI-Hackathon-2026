#!/usr/bin/env python3
"""
Script to update the synthetic dataset:
1. Move PatientID to first column
2. Change PatientID format to P00001, P00002, etc.
3. Convert Age and SpO2 to integers
4. Replace "Transgender" with "T"
"""

import pandas as pd

def update_dataset(input_file, output_file):
    """Update the dataset with required changes."""
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # 1. Replace "Transgender" with "T" in Gender column
    print("\nReplacing 'Transgender' with 'T'...")
    df['Gender'] = df['Gender'].replace('Transgender', 'T')
    transgender_count = (df['Gender'] == 'T').sum()
    print(f"  Found {transgender_count} records with Gender='T'")
    
    # 2. Convert Age and SpO2 to integers
    print("\nConverting Age and SpO2 to integers...")
    # Check for NaN values first
    age_nan = df['Age'].isna().sum()
    spo2_nan = df['SpO2'].isna().sum()
    if age_nan > 0:
        print(f"  Warning: {age_nan} NaN values in Age, filling with median...")
        df['Age'] = df['Age'].fillna(df['Age'].median())
    if spo2_nan > 0:
        print(f"  Warning: {spo2_nan} NaN values in SpO2, filling with median...")
        df['SpO2'] = df['SpO2'].fillna(df['SpO2'].median())
    
    # Convert to int (round first to handle any float precision issues)
    df['Age'] = df['Age'].round().astype(int)
    df['SpO2'] = df['SpO2'].round().astype(int)
    print("  ✓ Age and SpO2 converted to integers")
    
    # 3. Update PatientID format to P00001, P00002, etc.
    print("\nUpdating PatientID format...")
    df['PatientID'] = [f"P{i+1:05d}" for i in range(len(df))]
    print(f"  ✓ PatientID updated to P00001-P{len(df):05d} format")
    
    # 4. Reorder columns to put PatientID first
    print("\nReordering columns...")
    cols = df.columns.tolist()
    cols.remove('PatientID')
    cols.insert(0, 'PatientID')
    df = df[cols]
    print(f"  ✓ PatientID moved to first column")
    print(f"  Column order: {', '.join(cols[:5])}...")
    
    # Save the updated dataset
    print(f"\nSaving updated dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"✓ Successfully saved {len(df)} rows to {output_file}")
    
    # Display sample
    print("\nSample of updated data:")
    print(df.head(10).to_string())
    
    return df

if __name__ == "__main__":
    input_file = "triage_synthetic_dataset_20k.csv"
    output_file = "triage_synthetic_dataset_20k.csv"  # Overwrite the same file
    
    df = update_dataset(input_file, output_file)
    print("\n✓ Dataset update completed!")
