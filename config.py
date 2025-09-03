#!/usr/bin/env python3
"""
Configuration file for Claims Cost Prediction project.
Contains all configurable parameters and settings.
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, 'notebooks')

# Data configuration
DATA_CONFIG = {
    'target_column': 'charges',
    'categorical_columns': ['sex', 'smoker', 'region'],
    'numeric_columns': ['age', 'bmi', 'children'],
    'test_size': 0.2,
    'validation_size': 0.2,
    'random_state': 42,
    'batch_size': 32
}

# Model configuration
MODEL_CONFIG = {
    'linear': {
        'learning_rate': 0.001,
        'weight_decay': 1e-5
    },
    'mlp': {
        'hidden_sizes': [128, 64, 32],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'weight_decay': 1e-5
    },
    'deep_mlp': {
        'hidden_sizes': [256, 128, 128, 64, 64, 32],
        'dropout_rate': 0.3,
        'use_residual': True,
        'learning_rate': 0.001,
        'weight_decay': 1e-5
    },
    'attention_mlp': {
        'hidden_sizes': [128, 64],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'weight_decay': 1e-5
    }
}

# Training configuration
TRAINING_CONFIG = {
    'num_epochs': 100,
    'patience': 15,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'batch_size': 32,
    'device': 'auto'  # 'auto', 'cpu', or 'cuda'
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 100,
    'style': 'default',
    'color_palette': 'husl',
    'save_format': 'png',
    'interactive_backend': 'plotly'
}

# Business insights configuration
INSIGHTS_CONFIG = {
    'age_groups': [0, 25, 35, 45, 55, 100],
    'age_labels': ['18-25', '26-35', '36-45', '46-55', '55+'],
    'bmi_categories': [0, 18.5, 25, 30, 100],
    'bmi_labels': ['Underweight', 'Normal', 'Overweight', 'Obese'],
    'cost_thresholds': {
        'low': 5000,
        'medium': 15000,
        'high': 30000
    }
}

# File paths
FILE_PATHS = {
    'sample_data': os.path.join(DATA_DIR, 'medical_insurance_sample.csv'),
    'clean_data': os.path.join(DATA_DIR, 'medical_insurance_clean.csv'),
    'feature_info': os.path.join(RESULTS_DIR, 'feature_info.csv'),
    'correlations': os.path.join(RESULTS_DIR, 'feature_correlations.csv'),
    'business_insights': os.path.join(RESULTS_DIR, 'business_insights.csv'),
    'model_comparison': os.path.join(RESULTS_DIR, 'model_comparison.csv'),
    'interactive_dashboard': os.path.join(RESULTS_DIR, 'interactive_dashboard.html')
}

# Kaggle API configuration (if using real dataset)
KAGGLE_CONFIG = {
    'dataset_id': 'madhurpant/medical-insurance-cost-prediction',
    'api_path': os.path.expanduser('~/.kaggle/kaggle.json')
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.path.join(RESULTS_DIR, 'project.log')
}

# Performance metrics
METRICS = {
    'primary': 'r2',
    'secondary': ['mae', 'mse', 'rmse'],
    'thresholds': {
        'excellent_r2': 0.8,
        'good_r2': 0.6,
        'acceptable_r2': 0.4
    }
}

# Feature engineering options
FEATURE_ENGINEERING = {
    'create_age_groups': True,
    'create_bmi_categories': True,
    'create_interaction_features': False,
    'polynomial_features': False,
    'feature_scaling': True
}

# Model saving configuration
MODEL_SAVING = {
    'save_format': 'pth',
    'include_optimizer': False,
    'include_training_history': True,
    'save_feature_importance': True
}

def get_config():
    """Get complete configuration dictionary."""
    return {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'insights': INSIGHTS_CONFIG,
        'file_paths': FILE_PATHS,
        'kaggle': KAGGLE_CONFIG,
        'logging': LOGGING_CONFIG,
        'metrics': METRICS,
        'feature_engineering': FEATURE_ENGINEERING,
        'model_saving': MODEL_SAVING
    }

def create_directories():
    """Create necessary project directories."""
    directories = [DATA_DIR, RESULTS_DIR, MODELS_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def validate_config():
    """Validate configuration settings."""
    errors = []
    
    # Check if required directories exist
    if not os.path.exists(PROJECT_ROOT):
        errors.append(f"Project root directory not found: {PROJECT_ROOT}")
    
    # Check if source directory exists
    if not os.path.exists(SRC_DIR):
        errors.append(f"Source directory not found: {SRC_DIR}")
    
    # Validate numeric ranges
    if DATA_CONFIG['test_size'] + DATA_CONFIG['validation_size'] >= 1.0:
        errors.append("Test size + validation size must be less than 1.0")
    
    if TRAINING_CONFIG['num_epochs'] <= 0:
        errors.append("Number of epochs must be positive")
    
    if TRAINING_CONFIG['patience'] <= 0:
        errors.append("Patience must be positive")
    
    if len(errors) == 0:
        print("✅ Configuration validation passed")
        return True
    else:
        print("❌ Configuration validation failed:")
        for error in errors:
            print(f"   • {error}")
        return False

if __name__ == "__main__":
    print("🔧 Claims Cost Prediction Project Configuration")
    print("="*50)
    
    # Create directories
    create_directories()
    
    # Validate configuration
    if validate_config():
        print("\n🎉 Configuration is ready!")
        print("\n📋 Configuration summary:")
        config = get_config()
        for section, settings in config.items():
            print(f"   {section}: {len(settings)} settings")
    else:
        print("\n❌ Configuration has errors. Please fix them before proceeding.")
