#!/usr/bin/env python3
"""
PyTorch model definitions for Medical Cost prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LinearRegressionModel(nn.Module):
    """Simple linear regression model using PyTorch."""
    
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        
    def forward(self, x):
        return self.linear(x)

class MLPModel(nn.Module):
    """Multi-Layer Perceptron for Medical Cost prediction."""
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.2):
        super(MLPModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

class DeepMLPModel(nn.Module):
    """Deep MLP with residual connections and advanced architecture."""
    
    def __init__(self, input_size, hidden_sizes=[256, 128, 128, 64, 64, 32], 
                 dropout_rate=0.3, use_residual=True):
        super(DeepMLPModel, self).__init__()
        
        self.use_residual = use_residual
        self.input_size = input_size
        
        # Input projection layer
        self.input_projection = nn.Linear(input_size, hidden_sizes[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            layer = nn.Sequential(
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                nn.BatchNorm1d(hidden_sizes[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.hidden_layers.append(layer)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Input projection
        x = F.relu(self.input_projection(x))
        
        # Hidden layers with residual connections
        for i, layer in enumerate(self.hidden_layers):
            if self.use_residual and i < len(self.hidden_layers) - 1:
                # Residual connection (only if dimensions match)
                residual = x
                x = layer(x)
                if x.shape[1] == residual.shape[1]:
                    x = x + residual
            else:
                x = layer(x)
        
        # Output
        return self.output_layer(x)

class AttentionMLPModel(nn.Module):
    """MLP with attention mechanism for feature importance."""
    
    def __init__(self, input_size, hidden_sizes=[128, 64], dropout_rate=0.2):
        super(AttentionMLPModel, self).__init__()
        
        self.input_size = input_size
        
        # Feature attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size),
            nn.Sigmoid()
        )
        
        # Main network
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Apply attention
        attention_weights = self.attention(x)
        x_attended = x * attention_weights
        
        # Pass through main network
        return self.network(x_attended), attention_weights

def get_model(model_type, input_size, **kwargs):
    """Factory function to create models."""
    
    if model_type.lower() == 'linear':
        return LinearRegressionModel(input_size)
    
    elif model_type.lower() == 'mlp':
        hidden_sizes = kwargs.get('hidden_sizes', [128, 64, 32])
        dropout_rate = kwargs.get('dropout_rate', 0.2)
        return MLPModel(input_size, hidden_sizes, dropout_rate)
    
    elif model_type.lower() == 'deep_mlp':
        hidden_sizes = kwargs.get('hidden_sizes', [256, 128, 128, 64, 64, 32])
        dropout_rate = kwargs.get('dropout_rate', 0.3)
        use_residual = kwargs.get('use_residual', True)
        return DeepMLPModel(input_size, hidden_sizes, dropout_rate, use_residual)
    
    elif model_type.lower() == 'attention_mlp':
        hidden_sizes = kwargs.get('hidden_sizes', [128, 64])
        dropout_rate = kwargs.get('dropout_rate', 0.2)
        return AttentionMLPModel(input_size, hidden_sizes, dropout_rate)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_summary(model, input_size):
    """Get a summary of the model architecture."""
    summary = {
        'model_type': model.__class__.__name__,
        'input_size': input_size,
        'total_parameters': count_parameters(model),
        'trainable_parameters': count_parameters(model)
    }
    
    # Add model-specific information
    if hasattr(model, 'network'):
        if isinstance(model.network, nn.Sequential):
            summary['layers'] = []
            for i, layer in enumerate(model.network):
                if isinstance(layer, nn.Linear):
                    summary['layers'].append({
                        'type': 'Linear',
                        'in_features': layer.in_features,
                        'out_features': layer.out_features
                    })
                elif isinstance(layer, nn.BatchNorm1d):
                    summary['layers'].append({
                        'type': 'BatchNorm1d',
                        'num_features': layer.num_features
                    })
                elif isinstance(layer, nn.Dropout):
                    summary['layers'].append({
                        'type': 'Dropout',
                        'p': layer.p
                    })
                elif isinstance(layer, nn.ReLU):
                    summary['layers'].append({
                        'type': 'ReLU'
                    })
    
    return summary

def save_model(model, filepath):
    """Save a PyTorch model."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'model_config': get_model_summary(model, model.input_size if hasattr(model, 'input_size') else None)
    }, filepath)

def load_model(filepath, model_class, input_size):
    """Load a PyTorch model."""
    checkpoint = torch.load(filepath, map_location='cpu')
    model = model_class(input_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

if __name__ == "__main__":
    # Test model creation
    input_size = 6  # Example for medical insurance features
    
    print("🧪 Testing model creation...")
    
    # Test Linear Regression
    linear_model = get_model('linear', input_size)
    print(f"✅ Linear Model: {count_parameters(linear_model)} parameters")
    
    # Test MLP
    mlp_model = get_model('mlp', input_size)
    print(f"✅ MLP Model: {count_parameters(mlp_model)} parameters")
    
    # Test Deep MLP
    deep_mlp_model = get_model('deep_mlp', input_size)
    print(f"✅ Deep MLP Model: {count_parameters(deep_mlp_model)} parameters")
    
    # Test Attention MLP
    attention_mlp_model = get_model('attention_mlp', input_size)
    print(f"✅ Attention MLP Model: {count_parameters(attention_mlp_model)} parameters")
    
    print("\n🎉 All models created successfully!")
