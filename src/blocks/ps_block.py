import torch
import torch.nn as nn


class PSBlock(nn.Module):
    """
    Parameter Shared Block implementing Equation 3 from PSformer paper:
    Xout = (GeLU(XinW(1))W(2) + Xin)W(3)
    """
    
    def __init__(self, N: int):
        """
        Args:
            N: Dimension size for N×N weight matrices
        """
        super().__init__()
        self.N = N
        
        # Three N×N linear layers with bias
        self.linear1 = nn.Linear(N, N)
        self.linear2 = nn.Linear(N, N) 
        self.linear3 = nn.Linear(N, N)
        
        # Activation function
        self.activation = nn.GELU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization for W1, W2 and smaller weights for W3"""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        # Initialize linear3 with smaller weights as it's the final transformation
        nn.init.xavier_uniform_(self.linear3.weight, gain=0.1)
        
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)
        if self.linear3.bias is not None:
            nn.init.zeros_(self.linear3.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing the three-step transformation
        
        Args:
            x: Input tensor of shape (C, N)
            
        Returns:
            Output tensor of shape (C, N)
        """
        # Validate input shape
        if x.dim() != 2:
            raise ValueError(f"Input tensor must be 2-dimensional, got {x.dim()}")
            
        if x.shape[1] != self.N:
            raise ValueError(f"Input tensor second dimension must be {self.N}, got {x.shape[1]}")
            
        # Check for NaN or infinite values
        if torch.isnan(x).any():
            raise ValueError("Input tensor contains NaN values")
            
        if torch.isinf(x).any():
            raise ValueError("Input tensor contains infinite values")
        
        # Store original input for residual connection
        residual = x
        
        # First transformation: Linear -> GeLU -> Linear + Residual
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        intermediate_output = x + residual
        
        # Second transformation: Linear
        final_output = self.linear3(intermediate_output)
        
        return final_output