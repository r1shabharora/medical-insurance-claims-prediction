#!/usr/bin/env python3
"""
Main execution script for Claims Cost Prediction project.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append('src')

from data_loader import DataProcessor
from models import get_model, count_parameters
from trainer import ModelTrainer, ModelComparator
from utils import save_all_analyses

def main():
    """Main execution function."""
    print("🚀 Starting Claims Cost Prediction Project")
    print("="*60)
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Data Processing
    print("\n📊 Step 1: Data Processing")
    print("-" * 40)
    
    try:
        processor = DataProcessor()
        
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
        
        print("✅ Data processing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during data processing: {e}")
        print("🔄 Creating sample data for development...")
        
        try:
            # Create sample data if real data is not available
            from src.download_data import create_sample_data
            df = create_sample_data()
            
            # Continue with sample data
            processor = DataProcessor('data/medical_insurance_sample.csv')
            X, y = processor.preprocess_data(df)
            X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
            X_train_scaled, X_val_scaled, X_test_scaled = processor.scale_features(
                X_train, X_val, X_test
            )
            train_loader, val_loader, test_loader = processor.create_data_loaders(
                X_train_scaled, X_val_scaled, X_test_scaled,
                y_train, y_val, y_test
            )
            feature_info = processor.save_processed_data(
                X_train_scaled, X_val_scaled, X_test_scaled,
                y_train, y_val, y_test
            )
            processor.get_feature_importance_data(X_train_scaled, y_train)
            
        except Exception as sample_error:
            print(f"❌ Failed to create sample data: {sample_error}")
            print("🔄 Creating minimal sample data manually...")
            
            # Create minimal data manually as last resort
            import numpy as np
            np.random.seed(42)
            n_samples = 100
            
            # Create minimal dataset
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
            
            # Save minimal data
            os.makedirs('data', exist_ok=True)
            df.to_csv('data/medical_insurance_minimal.csv', index=False)
            
            # Continue with minimal data
            processor = DataProcessor('data/medical_insurance_minimal.csv')
            X, y = processor.preprocess_data(df)
            X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
            X_train_scaled, X_val_scaled, X_test_scaled = processor.scale_features(
                X_train, X_val, X_test
            )
            train_loader, val_loader, test_loader = processor.create_data_loaders(
                X_train_scaled, X_val_scaled, X_test_scaled,
                y_train, y_val, y_test
            )
            feature_info = processor.save_processed_data(
                X_train_scaled, X_val_scaled, X_test_scaled,
                y_train, y_val, y_test
            )
            processor.get_feature_importance_data(X_train_scaled, y_train)
    
    # Step 2: Model Creation
    print("\n🤖 Step 2: Model Creation")
    print("-" * 40)
    
    input_size = feature_info['n_features']
    print(f"📏 Input features: {input_size}")
    
    # Create models
    models = {
        'Linear Regression': get_model('linear', input_size),
        'MLP (3 layers)': get_model('mlp', input_size, hidden_sizes=[128, 64, 32]),
        'Deep MLP': get_model('deep_mlp', input_size),
        'Attention MLP': get_model('attention_mlp', input_size)
    }
    
    # Display model information
    print("\n📋 Model Information:")
    for name, model in models.items():
        param_count = count_parameters(model)
        print(f"   {name}: {param_count:,} parameters")
    
    # Step 3: Model Training and Comparison
    print("\n🚀 Step 3: Model Training and Comparison")
    print("-" * 40)
    
    try:
        # Create model comparator
        comparator = ModelComparator(models, train_loader, val_loader, test_loader)
        
        # Train all models
        comparator.train_all_models(num_epochs=100, patience=15)
        
        # Compare models
        comparison_df = comparator.compare_models()
        
        # Plot comparison
        comparator.plot_comparison('results/model_comparison.png')
        
        # Save all results
        comparator.save_results()
        
        print("✅ Model training and comparison completed!")
        
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        print("🔄 Training individual models...")
        
        # Train models individually if comparison fails
        for name, model in models.items():
            try:
                print(f"\n🔄 Training {name}...")
                trainer = ModelTrainer(model, train_loader, val_loader, test_loader)
                
                # Train model
                train_losses, val_losses, train_metrics, val_metrics = trainer.train(
                    num_epochs=50, patience=10
                )
                
                # Evaluate on test set
                test_metrics, test_predictions, test_targets = trainer.evaluate(test_loader)
                
                print(f"✅ {name} training completed!")
                print(f"   Test MSE: {test_metrics['mse']:.2f}")
                print(f"   Test MAE: {test_metrics['mae']:.2f}")
                print(f"   Test R²: {test_metrics['r2']:.4f}")
                
                # Save model
                model_path = f'models/{name.lower().replace(" ", "_")}.pth'
                trainer.save_best_model(model_path)
                
                # Save training history
                history_df = pd.DataFrame({
                    'epoch': range(1, len(train_losses) + 1),
                    'train_loss': train_losses,
                    'val_loss': val_losses,
                    'train_mae': [m['mae'] for m in train_metrics],
                    'val_mae': [m['mae'] for m in val_metrics],
                    'train_r2': [m['r2'] for m in train_metrics],
                    'val_r2': [m['r2'] for m in val_metrics]
                })
                history_path = f'results/{name.lower().replace(" ", "_")}_training_history.csv'
                history_df.to_csv(history_path, index=False)
                
                # Save predictions
                pred_df = pd.DataFrame({
                    'actual': test_targets,
                    'predicted': test_predictions,
                    'error': test_targets - test_predictions
                })
                pred_path = f'results/{name.lower().replace(" ", "_")}_predictions.csv'
                pred_df.to_csv(pred_path, index=False)
                
            except Exception as model_error:
                print(f"❌ Error training {name}: {model_error}")
    
    # Step 4: Data Analysis and Business Insights
    print("\n📊 Step 4: Data Analysis and Business Insights")
    print("-" * 40)
    
    try:
        # Load original data for analysis
        if os.path.exists('data/medical_insurance_clean.csv'):
            df = pd.read_csv('data/medical_insurance_clean.csv')
        elif os.path.exists('data/medical_insurance_sample.csv'):
            df = pd.read_csv('data/medical_insurance_sample.csv')
        else:
            print("⚠️  No data file found for analysis")
            return
        
        # Generate all analyses
        analysis_paths = save_all_analyses(df, 'results')
        
        print("✅ Data analysis and business insights completed!")
        print(f"📁 All results saved to results/ directory")
        
    except Exception as e:
        print(f"❌ Error during data analysis: {e}")
    
    # Step 5: Summary Report
    print("\n📋 Step 5: Summary Report")
    print("-" * 40)
    
    print("\n🎉 PROJECT COMPLETION SUMMARY")
    print("="*60)
    
    # Check what was created
    results_dir = 'results'
    if os.path.exists(results_dir):
        files = os.listdir(results_dir)
        print(f"📁 Results directory contains {len(files)} files:")
        for file in sorted(files):
            print(f"   • {file}")
    
    models_dir = 'models'
    if os.path.exists(models_dir):
        model_files = os.listdir(models_dir)
        print(f"\n🤖 Trained models ({len(model_files)}):")
        for model_file in sorted(model_files):
            print(f"   • {model_file}")
    
    data_dir = 'data'
    if os.path.exists(data_dir):
        data_files = os.listdir(data_dir)
        print(f"\n📊 Data files ({len(data_files)}):")
        for data_file in sorted(data_files):
            print(f"   • {data_file}")
    
    print("\n🚀 Next Steps:")
    print("   1. Review results in the 'results/' directory")
    print("   2. Analyze model performance in CSV files")
    print("   3. Use interactive dashboard for exploration")
    print("   4. Apply insights to business decisions")
    
    print("\n🎯 Business Value:")
    print("   • Risk assessment for insurance pricing")
    print("   • Customer segmentation based on risk factors")
    print("   • Resource allocation for claims management")
    print("   • Predictive modeling for future liabilities")
    
    print(f"\n⏰ Project completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == "__main__":
    main()
