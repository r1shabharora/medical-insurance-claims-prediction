#!/usr/bin/env python3
"""
Data download script for Medical Cost dataset from Kaggle.
"""

import os
import zipfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    """Download the Medical Cost dataset from Kaggle."""
    
    # Initialize Kaggle API
    try:
        api = KaggleApi()
        api.authenticate()
        print("✅ Kaggle API authenticated successfully")
    except Exception as e:
        print(f"❌ Kaggle API authentication failed: {e}")
        print("Please ensure you have kaggle.json in ~/.kaggle/ directory")
        return False
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Download dataset
    try:
        print("📥 Downloading Medical Cost dataset...")
        api.dataset_download_files(
            'madhurpant/medical-insurance-cost-prediction',
            path='data',
            unzip=True
        )
        print("✅ Dataset downloaded successfully")
        
        # Check if the main CSV file exists
        csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]
        if csv_files:
            print(f"📊 Found CSV files: {csv_files}")
            
            # Load and display basic info about the dataset
            main_csv = csv_files[0]
            df = pd.read_csv(f'data/{main_csv}')
            print(f"\n📈 Dataset Info:")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Sample data:")
            print(df.head())
            
            # Save a clean version for analysis
            clean_csv_path = 'data/medical_insurance_clean.csv'
            df.to_csv(clean_csv_path, index=False)
            print(f"\n💾 Clean dataset saved to: {clean_csv_path}")
            
        return True
        
    except Exception as e:
        print(f"❌ Dataset download failed: {e}")
        return False

def create_sample_data():
    """Create sample data if Kaggle download fails."""
    
    print("🔄 Creating sample dataset for development...")
    
    import numpy as np
    
    # Generate synthetic data similar to Medical Cost dataset
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(18, 65, n_samples),
        'sex': np.random.choice(['male', 'female'], n_samples),
        'bmi': np.random.normal(26.5, 5.5, n_samples),
        'children': np.random.randint(0, 6, n_samples),
        'smoker': np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8]),
        'region': np.random.choice(['southwest', 'southeast', 'northwest', 'northeast'], n_samples),
        'charges': np.random.normal(13000, 12000, n_samples)
    }
    
    # Ensure positive charges
    data['charges'] = np.abs(data['charges'])
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    sample_path = 'data/medical_insurance_sample.csv'
    df.to_csv(sample_path, index=False)
    print(f"✅ Sample dataset created: {sample_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    return df

if __name__ == "__main__":
    print("🚀 Starting Medical Cost dataset download...")
    
    # Try to download from Kaggle first
    try:
        if download_dataset():
            print("🎉 Dataset setup completed successfully!")
        else:
            print("⚠️  Kaggle download failed, creating sample data...")
            create_sample_data()
            print("🎉 Sample dataset created for development!")
    except Exception as e:
        print(f"⚠️  Error during dataset setup: {e}")
        print("🔄 Creating sample data for development...")
        try:
            create_sample_data()
            print("🎉 Sample dataset created for development!")
        except Exception as sample_error:
            print(f"❌ Failed to create sample data: {sample_error}")
            print("Please check the error and try again.")
