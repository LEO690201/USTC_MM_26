"""
Deep Ritz method for solving PDEs.
Neural network architecture with ResNet blocks.
"""

import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    """ResNet block with skip connection."""
    
    def __init__(self, hidden_dim):
        super(ResNetBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        identity = x
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        out = out + identity
        out = self.relu(out)
        return out


class DeepRitzNetwork(nn.Module):
    """
    Deep Ritz network for solving Poisson equation.
    Uses ResNet blocks for improved gradient flow.
    """
    
    def __init__(self, input_dim=2, hidden_dim=64, num_blocks=4, output_dim=1):
        """
        Args:
            input_dim: Input dimension (2 for 2D spatial coordinates)
            hidden_dim: Hidden dimension
            num_blocks: Number of ResNet blocks
            output_dim: Output dimension (1 for scalar function)
        """
        super(DeepRitzNetwork, self).__init__()
        
        # Input layer: map from input_dim to hidden_dim
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_relu = nn.ReLU()
        
        # ResNet blocks
        self.res_blocks = nn.ModuleList([
            ResNetBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            u: Output tensor of shape (batch_size, output_dim)
        """
        out = self.input_relu(self.input_layer(x))
        
        for block in self.res_blocks:
            out = block(out)
        
        out = self.output_layer(out)
        return out


class DeepRitzNetworkNoResNet(nn.Module):
    """
    Deep Ritz network without ResNet blocks (for comparison).
    Simple fully connected network with activation functions.
    """
    
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=5, output_dim=1):
        """
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_layers: Number of hidden layers
            output_dim: Output dimension
        """
        super(DeepRitzNetworkNoResNet, self).__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
