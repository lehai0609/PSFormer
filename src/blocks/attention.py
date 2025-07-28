import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .ps_block import PSBlock


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism as described in the Attention Is All You Need paper.
    Implements: Attention(Q, K, V) = softmax(QK^T / sqrt(dk)) * V
    """
    
    def __init__(self, dropout_rate: float = 0.0):
        """
        Args:
            dropout_rate: Dropout rate for attention weights
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.scale = None
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query tensor of shape [batch, num_queries, dk]
            K: Key tensor of shape [batch, num_keys, dk]
            V: Value tensor of shape [batch, num_keys, dv]
            
        Returns:
            Tuple of (output, attention_weights) where:
            - output: Tensor of shape [batch, num_queries, dv]
            - attention_weights: Tensor of shape [batch, num_queries, num_keys]
        """
        # Validate input dimensions
        if Q.dim() != 3 or K.dim() != 3 or V.dim() != 3:
            raise ValueError("Q, K, and V must all be 3-dimensional tensors")
        
        if Q.shape[0] != K.shape[0] or Q.shape[0] != V.shape[0]:
            raise ValueError("Batch dimensions of Q, K, and V must match")
            
        if K.shape[1] != V.shape[1]:
            raise ValueError("Number of keys in K must match number of values in V")
            
        if Q.shape[2] != K.shape[2]:
            raise ValueError("Embedding dimensions of Q and K must match")
        
        # Get the embedding dimension
        dk = K.shape[2]
        self.scale = torch.sqrt(torch.tensor(dk, dtype=torch.float32, device=Q.device))
        
        # Compute attention scores: Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, num_queries, num_keys]
        
        # Scale the scores
        scaled_scores = scores / self.scale
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scaled_scores, dim=-1)
        
        # Apply dropout if specified
        if self.dropout is not None:
            attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # [batch, num_queries, dv]
        
        return output, attention_weights


class SegmentAttentionStage(nn.Module):
    """
    Single stage of segment attention using a shared PS Block to generate Q, K, V.
    """
    
    def __init__(self, ps_block: PSBlock, use_dropout: bool = False):
        """
        Args:
            ps_block: Shared PS Block used to generate Q, K, V
            use_dropout: Whether to use dropout in the attention mechanism
        """
        super().__init__()
        self.ps_block = ps_block
        self.attention = ScaledDotProductAttention(dropout_rate=0.1 if use_dropout else 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the segment attention stage.
        
        Args:
            x: Input tensor of shape [batch, N, C] where:
               - N is the number of segments (sequence dimension for attention)
               - C is the segment length (feature dimension)
               
        Returns:
            Tuple of (attention_output, attention_weights) where:
            - attention_output: Tensor of shape [batch, N, C]
            - attention_weights: Tensor of shape [batch, N, N]
        """
        # Validate input shape
        if x.dim() != 3:
            raise ValueError(f"Input tensor must be 3-dimensional, got {x.dim()}")
        
        batch, N, C = x.shape
        
        # Generate Q, K, V using the shared PS Block
        # In PSformer, all three come from the same PS Block output
        ps_output = self.ps_block(x)  # [batch, N, C]
        
        # Use same output for Q, K, V (key PSformer innovation)
        Q = ps_output
        K = ps_output
        V = ps_output
        
        # Apply attention
        attention_output, attention_weights = self.attention(Q, K, V)
        
        return attention_output, attention_weights


class TwoStageSegmentAttention(nn.Module):
    """
    Two-stage segment attention mechanism as described in the PSformer paper.
    """
    
    def __init__(self, ps_block: PSBlock):
        """
        Args:
            ps_block: Shared PS Block used across both attention stages
        """
        super().__init__()
        self.ps_block = ps_block  # Single shared PS Block
        self.stage1 = SegmentAttentionStage(ps_block)
        self.stage2 = SegmentAttentionStage(ps_block)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the two-stage segment attention.
        
        Args:
            x: Input tensor of shape [batch, N, C]
            
        Returns:
            Tuple of (output, (stage1_weights, stage2_weights)) where:
            - output: Tensor of shape [batch, N, C]
            - stage1_weights: Tensor of shape [batch, N, N]
            - stage2_weights: Tensor of shape [batch, N, N]
        """
        # Validate input shape
        if x.dim() != 3:
            raise ValueError(f"Input tensor must be 3-dimensional, got {x.dim()}")
        
        # Stage 1
        stage1_output, stage1_weights = self.stage1(x)
        
        # ReLU activation between stages
        activated_output = self.activation(stage1_output)
        
        # Stage 2
        stage2_output, stage2_weights = self.stage2(activated_output)
        
        return stage2_output, (stage1_weights, stage2_weights)


class PSformerEncoderLayer(nn.Module):
    """
    Single layer of the PSformer encoder.
    """
    
    def __init__(self, ps_block: PSBlock):
        """
        Args:
            ps_block: Shared PS Block used in all components of this layer
        """
        super().__init__()
        self.ps_block = ps_block  # Shared across all components
        self.two_stage_attention = TwoStageSegmentAttention(ps_block)
        self.final_ps_block = ps_block  # Same instance for final transformation
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the PSformer encoder layer.
        
        Args:
            x: Input tensor of shape [batch, N, C] from data transformer
            
        Returns:
            Tuple of (output, attention_weights) where:
            - output: Tensor of shape [batch, N, C]
            - attention_weights: Tuple of (stage1_weights, stage2_weights)
        """
        # Validate input shape
        if x.dim() != 3:
            raise ValueError(f"Input tensor must be 3-dimensional, got {x.dim()}")
        
        batch, N, C = x.shape
        
        # Two-stage attention
        attention_output, attention_weights = self.two_stage_attention(x)
        
        # Residual connection
        residual_output = attention_output + x
        
        # Final PS Block processing
        output = self.final_ps_block(residual_output)
        
        return output, attention_weights


class PSformerEncoder(nn.Module):
    """
    Complete PSformer encoder with multiple layers.
    """
    
    def __init__(self, num_layers: int, segment_length: int):
        """
        Args:
            num_layers: Number of encoder layers
            segment_length: Length of each segment (C = M * P)
        """
        super().__init__()
        # Each layer has its own PS Block
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            ps_block = PSBlock(N=segment_length)
            encoder_layer = PSformerEncoderLayer(ps_block)
            self.layers.append(encoder_layer)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through the PSformer encoder.
        
        Args:
            x: Input tensor of shape [batch, N, C] from data transformer
            
        Returns:
            Tuple of (output, attention_weights_list) where:
            - output: Tensor of shape [batch, N, C]
            - attention_weights_list: List of attention weights from each layer
        """
        # Validate input shape
        if x.dim() != 3:
            raise ValueError(f"Input tensor must be 3-dimensional, got {x.dim()}")
        
        attention_weights_list = []
        
        # Process through each layer
        for layer in self.layers:
            x, weights = layer(x)
            attention_weights_list.append(weights)
        
        return x, attention_weights_list