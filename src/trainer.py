#!/usr/bin/env python3
"""
Training utilities for PyTorch models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import json

class ModelTrainer:
    """Trainer class for PyTorch models."""
    
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 learning_rate=0.001, weight_decay=1e-5, device='auto'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"🚀 Using device: {self.device}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer and loss function
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_mae = 0
        total_r2 = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                mae = torch.mean(torch.abs(output - target))
                r2 = self._calculate_r2(output, target)
            
            total_loss += loss.item()
            total_mae += mae.item()
            total_r2 += r2
            num_batches += 1
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        avg_r2 = total_r2 / num_batches
        
        return avg_loss, avg_mae, avg_r2
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_r2 = 0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                mae = torch.mean(torch.abs(output - target))
                r2 = self._calculate_r2(output, target)
                
                total_loss += loss.item()
                total_mae += mae.item()
                total_r2 += r2
                num_batches += 1
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        avg_r2 = total_r2 / num_batches
        
        return avg_loss, avg_mae, avg_r2
    
    def _calculate_r2(self, y_pred, y_true):
        """Calculate R² score."""
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        return r2
    
    def train(self, num_epochs, patience=10, save_path=None):
        """Train the model with early stopping."""
        print(f"🚀 Starting training for {num_epochs} epochs...")
        print(f"📊 Training samples: {len(self.train_loader.dataset)}")
        print(f"📊 Validation samples: {len(self.val_loader.dataset)}")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_mae, train_r2 = self.train_epoch()
            
            # Validation
            val_loss, val_mae, val_r2 = self.validate_epoch()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append({'mae': train_mae, 'r2': train_r2})
            self.val_metrics.append({'mae': val_mae, 'r2': val_r2})
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Train MAE: {train_mae:.2f} | "
                      f"Val MAE: {val_mae:.2f} | "
                      f"Train R²: {train_r2:.4f} | "
                      f"Val R²: {val_r2:.4f}")
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                # Save best model
                if save_path:
                    self.save_best_model(save_path)
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"⏹️  Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"✅ Restored best model with validation loss: {self.best_val_loss:.4f}")
        
        return self.train_losses, self.val_losses, self.train_metrics, self.val_metrics
    
    def evaluate(self, data_loader):
        """Evaluate the model on a dataset."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                all_predictions.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                total_loss += loss.item()
                num_batches += 1
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions).flatten()
        targets = np.array(all_targets).flatten()
        
        # Calculate metrics
        mse = total_loss / num_batches
        mae = np.mean(np.abs(predictions - targets))
        r2 = self._calculate_r2_numpy(predictions, targets)
        rmse = np.sqrt(mse)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        return metrics, predictions, targets
    
    def _calculate_r2_numpy(self, y_pred, y_true):
        """Calculate R² score using numpy."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        return r2
    
    def save_best_model(self, save_path):
        """Save the best model."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.best_model_state,
            'model_config': self.model.__class__.__name__,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }, save_path)
    
    def plot_training_history(self, save_path=None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MSE)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE plot
        train_mae = [m['mae'] for m in self.train_metrics]
        val_mae = [m['mae'] for m in self.val_metrics]
        axes[0, 1].plot(train_mae, label='Train MAE', color='blue')
        axes[0, 1].plot(val_mae, label='Validation MAE', color='red')
        axes[0, 1].set_title('Training and Validation MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # R² plot
        train_r2 = [m['r2'] for m in self.train_metrics]
        val_r2 = [m['r2'] for m in self.val_metrics]
        axes[1, 0].plot(train_r2, label='Train R²', color='blue')
        axes[1, 0].plot(val_r2, label='Validation R²', color='red')
        axes[1, 0].set_title('Training and Validation R²')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Final metrics comparison
        final_train_mae = train_mae[-1] if train_mae else 0
        final_val_mae = val_mae[-1] if val_mae else 0
        final_train_r2 = train_r2[-1] if train_r2 else 0
        final_val_r2 = val_r2[-1] if val_r2 else 0
        
        metrics_text = f"""Final Metrics:
Train MAE: {final_train_mae:.2f}
Val MAE: {final_val_mae:.2f}
Train R²: {final_train_r2:.4f}
Val R²: {final_val_r2:.4f}"""
        
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 1].set_title('Final Model Performance')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training history plot saved to: {save_path}")
        
        plt.show()

class ModelComparator:
    """Compare multiple models."""
    
    def __init__(self, models_dict, train_loader, val_loader, test_loader):
        self.models = models_dict
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.results = {}
        
    def train_all_models(self, num_epochs=100, patience=15):
        """Train all models."""
        print("🚀 Training all models...")
        
        for model_name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training {model_name}...")
            print(f"{'='*50}")
            
            # Create trainer
            trainer = ModelTrainer(model, self.train_loader, self.val_loader, self.test_loader)
            
            # Train model
            train_losses, val_losses, train_metrics, val_metrics = trainer.train(
                num_epochs=num_epochs, patience=patience
            )
            
            # Evaluate on test set
            test_metrics, test_predictions, test_targets = trainer.evaluate(self.test_loader)
            
            # Store results
            self.results[model_name] = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'test_predictions': test_predictions,
                'test_targets': test_targets,
                'trainer': trainer
            }
            
            print(f"✅ {model_name} training completed!")
            print(f"   Test MSE: {test_metrics['mse']:.2f}")
            print(f"   Test MAE: {test_metrics['mae']:.2f}")
            print(f"   Test R²: {test_metrics['r2']:.4f}")
    
    def compare_models(self):
        """Compare all trained models."""
        print("\n📊 Model Comparison Results:")
        print("="*80)
        
        comparison_data = []
        
        for model_name, result in self.results.items():
            test_metrics = result['test_metrics']
            comparison_data.append({
                'Model': model_name,
                'MSE': test_metrics['mse'],
                'MAE': test_metrics['mae'],
                'RMSE': test_metrics['rmse'],
                'R²': test_metrics['r2']
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('R²', ascending=False)
        
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Save comparison results
        os.makedirs('results', exist_ok=True)
        comparison_df.to_csv('results/model_comparison.csv', index=False)
        
        return comparison_df
    
    def plot_comparison(self, save_path=None):
        """Plot comparison of all models."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['MSE', 'MAE', 'RMSE', 'R²']
        metric_names = ['Mean Squared Error', 'Mean Absolute Error', 'Root Mean Squared Error', 'R² Score']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            row, col = i // 2, i % 2
            
            values = [self.results[model]['test_metrics'][metric.lower().replace('²', '2')] 
                     for model in self.results.keys()]
            models = list(self.results.keys())
            
            bars = axes[row, col].bar(models, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
            axes[row, col].set_title(f'{metric_name} Comparison')
            axes[row, col].set_ylabel(metric)
            axes[row, col].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Model comparison plot saved to: {save_path}")
        
        plt.show()
    
    def save_results(self):
        """Save all results to CSV files."""
        os.makedirs('results', exist_ok=True)
        
        # Save detailed results for each model
        for model_name, result in self.results.items():
            # Save predictions vs targets
            pred_df = pd.DataFrame({
                'actual': result['test_targets'],
                'predicted': result['test_predictions'],
                'error': result['test_targets'] - result['test_predictions']
            })
            pred_df.to_csv(f'results/{model_name}_predictions.csv', index=False)
            
            # Save training history
            history_df = pd.DataFrame({
                'epoch': range(1, len(result['train_losses']) + 1),
                'train_loss': result['train_losses'],
                'val_loss': result['val_losses'],
                'train_mae': [m['mae'] for m in result['train_metrics']],
                'val_mae': [m['mae'] for m in result['val_metrics']],
                'train_r2': [m['r2'] for m in result['train_metrics']],
                'val_r2': [m['r2'] for m in result['val_metrics']]
            })
            history_df.to_csv(f'results/{model_name}_training_history.csv', index=False)
        
        print("💾 All results saved to results/ directory")

if __name__ == "__main__":
    print("🧪 Testing trainer module...")
    print("✅ Trainer module loaded successfully!")
