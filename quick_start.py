#!/usr/bin/env python3
"""
Quick start script for Claims Cost Prediction project.
This script provides a streamlined way to get started with the project.
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def create_sample_data():
    """Create sample data for immediate use."""
    print("🔄 Creating sample data...")
    try:
        sys.path.append('src')
        from download_data import create_sample_data
        
        df = create_sample_data()
        print(f"✅ Sample data created: {df.shape}")
        return True
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        print("🔄 Creating minimal sample data manually...")
        
        try:
            # Create minimal data manually as fallback
            import numpy as np
            import pandas as pd
            
            np.random.seed(42)
            n_samples = 100
            
            data = {
                'age': np.random.randint(18, 65, n_samples),
                'sex': np.random.choice(['male', 'female'], n_samples),
                'bmi': np.random.normal(26.5, 5.5, n_samples),
                'children': np.random.randint(0, 6, n_samples),
                'smoker': np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8]),
                'region': np.random.choice(['southwest', 'southeast', 'northwest', 'northeast'], n_samples),
                'charges': np.abs(np.random.normal(13000, 12000, n_samples))
            }
            
            df = pd.DataFrame(data)
            
            # Ensure data directory exists
            os.makedirs('data', exist_ok=True)
            df.to_csv('data/medical_insurance_minimal.csv', index=False)
            
            print(f"✅ Minimal sample data created: {df.shape}")
            return True
            
        except Exception as fallback_error:
            print(f"❌ Failed to create minimal sample data: {fallback_error}")
            return False

def run_quick_analysis():
    """Run a quick analysis to demonstrate the project."""
    print("🚀 Running quick analysis...")
    try:
        # Import modules
        from data_loader import DataProcessor
        from models import get_model
        from utils import save_all_analyses
        
        # Try to load sample data, fallback to minimal data
        data_files = ['data/medical_insurance_sample.csv', 'data/medical_insurance_minimal.csv']
        df = None
        
        for data_file in data_files:
            if os.path.exists(data_file):
                try:
                    processor = DataProcessor(data_file)
                    df = processor.load_data()
                    print(f"✅ Loaded data from: {data_file}")
                    break
                except Exception as e:
                    print(f"⚠️  Failed to load {data_file}: {e}")
                    continue
        
        if df is None:
            print("❌ No data files found or could be loaded")
            return False
        
        # Quick analysis
        analysis_paths = save_all_analyses(df, 'results')
        
        print("✅ Quick analysis completed!")
        print("📁 Check the 'results/' directory for outputs")
        return True
        
    except Exception as e:
        print(f"❌ Quick analysis failed: {e}")
        return False

def main():
    """Main quick start function."""
    print("🚀 Claims Cost Prediction - Quick Start")
    print("="*50)
    
    # Step 1: Check Python version
    if not check_python_version():
        return
    
    # Step 2: Install requirements
    if not install_requirements():
        print("❌ Cannot continue without required packages")
        return
    
    # Step 3: Create sample data
    if not create_sample_data():
        print("❌ Cannot continue without data")
        return
    
    # Step 4: Run quick analysis
    if run_quick_analysis():
        print("\n🎉 Quick start completed successfully!")
        print("\n📋 What was created:")
        print("   • Sample dataset in data/ directory")
        print("   • Analysis results in results/ directory")
        print("   • Interactive dashboard (HTML file)")
        print("   • Business insights (CSV file)")
        
        print("\n🚀 Next steps:")
        print("   1. Explore results in the 'results/' directory")
        print("   2. Run full training: python main.py")
        print("   3. Use Jupyter notebooks for detailed analysis")
        print("   4. Customize models and parameters")
        
        print("\n💡 Tips:")
        print("   • Open the HTML dashboard in your browser")
        print("   • Check CSV files for detailed insights")
        print("   • Modify parameters in src/ files")
        print("   • Add your own data to the data/ directory")
    else:
        print("❌ Quick start failed. Check the errors above.")

if __name__ == "__main__":
    main()
