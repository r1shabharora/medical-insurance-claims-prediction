#!/usr/bin/env python3
"""
Data loading and preprocessing utilities for Medical Cost dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch.utils.data import Dataset, DataLoader
import os

class MedicalInsuranceDataset(Dataset):
    """PyTorch Dataset for Medical Insurance data."""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets).reshape(-1, 1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class DataProcessor:
    """Data preprocessing and loading utilities."""
    
    def __init__(self, data_path='data/medical_insurance_clean.csv'):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = 'charges'
        
    def load_data(self):
        """Load and display basic information about the dataset."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        print(f"📊 Dataset loaded: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"🎯 Target variable: {self.target_column}")
        
        # Display basic statistics
        print(f"\n📈 Basic Statistics:")
        print(df.describe())
        
        # Check for missing values
        print(f"\n🔍 Missing Values:")
        print(df.isnull().sum())
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for machine learning."""
        print("🔄 Preprocessing data...")
        
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Handle categorical variables
        categorical_columns = ['sex', 'smoker', 'region']
        
        for col in categorical_columns:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
                print(f"   Encoded {col}: {le.classes_}")
        
        # Separate features and target
        feature_columns = [col for col in df_processed.columns if col != self.target_column]
        self.feature_columns = feature_columns
        
        X = df_processed[feature_columns].values
        y = df_processed[self.target_column].values
        
        print(f"   Features: {feature_columns}")
        print(f"   Feature shape: {X.shape}")
        print(f"   Target shape: {y.shape}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """Split data into train, validation, and test sets."""
        print("✂️  Splitting data...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        print(f"   Train set: {X_train.shape[0]} samples")
        print(f"   Validation set: {X_val.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test):
        """Scale features using StandardScaler."""
        print("📏 Scaling features...")
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Transform validation and test data
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"   Features scaled using StandardScaler")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test, 
                           batch_size=32):
        """Create PyTorch DataLoaders."""
        print("📦 Creating PyTorch DataLoaders...")
        
        # Create datasets
        train_dataset = MedicalInsuranceDataset(X_train, y_train)
        val_dataset = MedicalInsuranceDataset(X_val, y_val)
        test_dataset = MedicalInsuranceDataset(X_test, y_test)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"   Batch size: {batch_size}")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Save processed data to CSV files for analysis."""
        print("💾 Saving processed data...")
        
        # Create results directory
        os.makedirs("results", exist_ok=True)
        
        # Save training data
        train_df = pd.DataFrame(X_train, columns=self.feature_columns)
        train_df['charges'] = y_train
        train_df.to_csv('results/train_data.csv', index=False)
        
        # Save validation data
        val_df = pd.DataFrame(X_val, columns=self.feature_columns)
        val_df['charges'] = y_val
        val_df.to_csv('results/val_data.csv', index=False)
        
        # Save test data
        test_df = pd.DataFrame(X_test, columns=self.feature_columns)
        test_df['charges'] = y_test
        test_df.to_csv('results/test_data.csv', index=False)
        
        # Save feature information
        feature_info = {
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'n_features': len(self.feature_columns),
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val),
            'n_test_samples': len(X_test)
        }
        
        feature_df = pd.DataFrame([feature_info])
        feature_df.to_csv('results/feature_info.csv', index=False)
        
        print(f"   ✅ Data saved to results/ directory")
        
        return feature_info
    
    def get_feature_importance_data(self, X_train, y_train):
        """Prepare data for feature importance analysis."""
        print("🔍 Preparing feature importance data...")
        
        # Calculate correlation with target
        correlations = {}
        for i, feature in enumerate(self.feature_columns):
            correlation = np.corrcoef(X_train[:, i], y_train)[0, 1]
            correlations[feature] = correlation
        
        # Create correlation DataFrame
        corr_df = pd.DataFrame(list(correlations.items()), 
                              columns=['feature', 'correlation'])
        corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)
        
        # Save correlation data
        corr_df.to_csv('results/feature_correlations.csv', index=False)
        
        print(f"   ✅ Feature correlations saved")
        print(f"   📊 Top correlations:")
        print(corr_df.head())
        
        return corr_df

def main():
    """Main function to demonstrate data processing."""
    processor = DataProcessor()
    
    try:
        # Load data
        df = processor.load_data()
        
        # Preprocess data
        X, y = processor.preprocess_data(df)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = processor.scale_features(
            X_train, X_val, X_test
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader = processor.create_data_loaders(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test
        )
        
        # Save processed data
        feature_info = processor.save_processed_data(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test
        )
        
        # Get feature importance data
        processor.get_feature_importance_data(X_train_scaled, y_train)
        
        print("\n🎉 Data processing completed successfully!")
        print(f"📁 Check 'results/' directory for processed data files")
        
        return train_loader, val_loader, test_loader, feature_info
        
    except Exception as e:
        print(f"❌ Error during data processing: {e}")
        return None, None, None, None

if __name__ == "__main__":
    main()
