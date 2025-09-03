#!/usr/bin/env python3
"""
Test script to verify project setup and dependencies.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("🧪 Testing package imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError:
        print("❌ Pandas not found")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.version.version}")
    except ImportError:
        print("❌ NumPy not found")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("❌ Scikit-learn not found")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("❌ Matplotlib not found")
        return False
    
    try:
        import seaborn as sns
        print(f"✅ Seaborn: {sns.__version__}")
    except ImportError:
        print("❌ Seaborn not found")
        return False
    
    try:
        import plotly
        print(f"✅ Plotly: {plotly.__version__}")
    except ImportError:
        print("❌ Plotly not found")
        return False
    
    return True

def test_src_modules():
    """Test if source modules can be imported."""
    print("\n🧪 Testing source modules...")
    
    # Add src to path
    sys.path.append('src')
    
    try:
        from data_loader import DataProcessor
        print("✅ DataProcessor imported successfully")
    except ImportError as e:
        print(f"❌ DataProcessor import failed: {e}")
        return False
    
    try:
        from models import get_model, count_parameters
        print("✅ Models module imported successfully")
    except ImportError as e:
        print(f"❌ Models module import failed: {e}")
        return False
    
    try:
        from trainer import ModelTrainer, ModelComparator
        print("✅ Trainer module imported successfully")
    except ImportError as e:
        print(f"❌ Trainer module import failed: {e}")
        return False
    
    try:
        from utils import save_all_analyses
        print("✅ Utils module imported successfully")
    except ImportError as e:
        print(f"❌ Utils module import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test if models can be created."""
    print("\n🧪 Testing model creation...")
    
    try:
        from models import get_model, count_parameters
        
        # Test linear model
        linear_model = get_model('linear', 6)
        param_count = count_parameters(linear_model)
        print(f"✅ Linear model created: {param_count} parameters")
        
        # Test MLP model
        mlp_model = get_model('mlp', 6)
        param_count = count_parameters(mlp_model)
        print(f"✅ MLP model created: {param_count} parameters")
        
        # Test deep MLP model
        deep_mlp_model = get_model('deep_mlp', 6)
        param_count = count_parameters(deep_mlp_model)
        print(f"✅ Deep MLP model created: {param_count} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_sample_data():
    """Test sample data creation."""
    print("\n🧪 Testing sample data creation...")
    
    try:
        from download_data import create_sample_data
        
        df = create_sample_data()
        print(f"✅ Sample data created: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Sample values:")
        print(df.head())
        
        return True
        
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Claims Cost Prediction Project - Setup Test")
    print("="*60)
    
    all_tests_passed = True
    
    # Test 1: Package imports
    if not test_imports():
        all_tests_passed = False
    
    # Test 2: Source modules
    if not test_src_modules():
        all_tests_passed = False
    
    # Test 3: Model creation
    if not test_model_creation():
        all_tests_passed = False
    
    # Test 4: Sample data
    if not test_sample_data():
        all_tests_passed = False
    
    # Summary
    print("\n" + "="*60)
    if all_tests_passed:
        print("🎉 All tests passed! Project setup is ready.")
        print("\n🚀 Next steps:")
        print("   1. Run: python main.py")
        print("   2. Or use Jupyter notebooks in notebooks/ directory")
        print("   3. Check results/ directory for outputs")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Install requirements: pip install -r requirements.txt")
        print("   2. Check Python version compatibility")
        print("   3. Verify all files are in correct locations")
    
    print("="*60)

if __name__ == "__main__":
    main()
