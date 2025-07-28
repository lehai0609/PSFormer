import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
from .RevIN import RevIN
from .data_transformer import PSformerDataTransformer, create_transformer_for_psformer, get_psformer_dimensions
from .attention import PSformerEncoder


class PSformerConfig:
    """
    Configuration class for PSformer model parameters
    """
    def __init__(self, 
                 sequence_length: int,
                 num_variables: int, 
                 patch_size: int,
                 num_encoder_layers: int,
                 prediction_length: int,
                 affine_revin: bool = True,
                 revin_eps: float = 1e-5):
        """
        Args:
            sequence_length: Total input sequence length (L)
            num_variables: Number of time series variables (M)
            patch_size: Size of each temporal patch (P)
            num_encoder_layers: Number of PSformer encoder layers
            prediction_length: Length of prediction horizon (F)
            affine_revin: Whether to use learnable affine parameters in RevIN
            revin_eps: Small value for numerical stability in RevIN
        """
        self.sequence_length = sequence_length
        self.num_variables = num_variables
        self.patch_size = patch_size
        self.num_encoder_layers = num_encoder_layers
        self.prediction_length = prediction_length
        self.affine_revin = affine_revin
        self.revin_eps = revin_eps
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters"""
        if self.sequence_length % self.patch_size != 0:
            raise ValueError(f"Sequence length {self.sequence_length} must be divisible by patch size {self.patch_size}")
        if self.num_variables <= 0:
            raise ValueError(f"Number of variables must be positive, got {self.num_variables}")
        if self.patch_size <= 0:
            raise ValueError(f"Patch size must be positive, got {self.patch_size}")
        if self.num_encoder_layers <= 0:
            raise ValueError(f"Number of encoder layers must be positive, got {self.num_encoder_layers}")
        if self.prediction_length <= 0:
            raise ValueError(f"Prediction length must be positive, got {self.prediction_length}")


class PSformer(nn.Module):
    """
    Main PSformer model implementing the complete input processing pipeline.
    
    Architecture: Raw Input → RevIN Normalization → Data Transformation → PSformer Encoder
    
    Based on the PSformer paper architecture described in Section 3.2 and Figures 1 & 2.
    """
    
    def __init__(self, config: PSformerConfig):
        """
        Initialize PSformer model with all components.
        
        Args:
            config: PSformerConfig instance containing model parameters
        """
        super().__init__()
        self.config = config
        
        # 1. Instantiate the Reversible Instance Normalization (RevIN) layer
        # It normalizes each variable independently
        self.revin_layer = RevIN(
            num_features=config.num_variables,
            eps=config.revin_eps,
            affine=config.affine_revin
        )
        
        # 2. Instantiate the Data Transformer
        # This handles patching and segment creation
        self.data_transformer = create_transformer_for_psformer(
            sequence_length=config.sequence_length,
            num_variables=config.num_variables,
            patch_size=config.patch_size
        )
        
        # 3. Get key dimensions from the data transformer
        # This ensures the encoder is built with correct C and N dimensions
        psformer_dims = get_psformer_dimensions(self.data_transformer)
        # C = segment_length = M * P
        # N = num_patches = L / P
        
        # 4. Instantiate the PSformer Encoder
        # The encoder uses the segment_length (C) as the feature dimension for attention
        self.encoder = PSformerEncoder(
            num_layers=config.num_encoder_layers,
            segment_length=psformer_dims['C']
        )
        
        # 5. Add the final linear projection layer for forecasting
        # This layer maps the sequence length (L) to the prediction length (F)
        # Paper reference: X_pred = X_out * W_F where W_F ∈ R^(L×F)
        self.output_projection = nn.Linear(
            in_features=config.sequence_length,
            out_features=config.prediction_length
        )
        
        # Store dimensions for easy access
        self.psformer_dims = psformer_dims
    
    def forward(self, raw_input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing the complete pipeline: 
        Input -> Normalization -> Transformation -> Encoder -> Inverse Transformation -> Projection -> Inverse Normalization -> Output
        
        Args:
            raw_input_tensor: Input tensor of shape [batch_size, num_variables, sequence_length]
                             or [batch, M, L] as per the paper notation
        
        Returns:
            final_predictions: Tensor of shape [batch, M, F] where F is prediction_length
        """
        # Validate input tensor
        self._validate_input(raw_input_tensor)
        
        # ----- START OF INPUT PROCESSING PIPELINE -----
        
        # STEP 1: NORMALIZATION
        # Apply RevIN layer in 'norm' mode to the raw input
        # This is the first step shown in Figure 1 and mentioned in Section 3.2
        # The statistics (mean, stdev) are automatically stored inside self.revin_layer
        normalized_input = self.revin_layer(raw_input_tensor, mode='norm')
        
        # STEP 2: DATA TRANSFORMATION (Patching & Segmenting)
        # Use the data_transformer's forward_transform method
        # This transforms the input from [batch, M, L] to [batch, N, C]
        encoder_ready_data = self.data_transformer.forward_transform(normalized_input)
        
        # STEP 3: ENCODER PROCESSING
        # Feed the prepared data into the encoder
        # The encoder returns its final output and a list of attention weights
        encoder_output, attention_weights_list = self.encoder(encoder_ready_data)
        
        # ----- END OF INPUT PROCESSING, START OF OUTPUT PIPELINE -----
        
        # STEP 4.1: INVERSE DATA TRANSFORMATION
        # Description: Convert the encoder's segmented output back to time series format.
        # Paper reference: "Inverse Transformation" block in Fig 1 & 2.
        reshaped_output = self.data_transformer.inverse_transform(
            encoder_output, 
            self.config.sequence_length
        )  # Shape: [B, M, L]
        
        # STEP 4.2: LINEAR PROJECTION FOR FORECASTING
        # Description: Project the restored sequence to the desired prediction horizon.
        # Paper reference: "Linear Mapping" (Fig 1) or WF matrix multiplication (Section 3.2).
        projected_output = self.output_projection(reshaped_output)  # Shape: [B, M, F]
        
        # STEP 4.3: INVERSE NORMALIZATION (RevIN Denorm)
        # Description: Scale the forecast back to the original data distribution.
        # Paper reference: "Inverse RevIN" (Fig 1) or "RevIN⁻¹" (Fig 2).
        final_predictions = self.revin_layer(projected_output, mode='denorm')  # Shape: [B, M, F]
        
        return final_predictions
    
    def forward_with_intermediates(self, raw_input_tensor: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass that returns both final predictions and intermediate outputs for analysis.
        
        Args:
            raw_input_tensor: Input tensor of shape [batch_size, num_variables, sequence_length]
        
        Returns:
            Tuple of (final_predictions, intermediates) where intermediates contains:
            - encoder_output: Output from encoder
            - attention_weights: List of attention weights 
            - reshaped_output: Output after inverse transformation
            - projected_output: Output after linear projection
        """
        # Validate input tensor
        self._validate_input(raw_input_tensor)
        
        # Input processing pipeline
        normalized_input = self.revin_layer(raw_input_tensor, mode='norm')
        encoder_ready_data = self.data_transformer.forward_transform(normalized_input)
        encoder_output, attention_weights_list = self.encoder(encoder_ready_data)
        
        # Output processing pipeline
        reshaped_output = self.data_transformer.inverse_transform(
            encoder_output, 
            self.config.sequence_length
        )
        projected_output = self.output_projection(reshaped_output)
        final_predictions = self.revin_layer(projected_output, mode='denorm')
        
        intermediates = {
            'encoder_output': encoder_output,
            'attention_weights': attention_weights_list,
            'reshaped_output': reshaped_output,
            'projected_output': projected_output
        }
        
        return final_predictions, intermediates
    
    def _validate_input(self, input_tensor: torch.Tensor):
        """
        Validate the input tensor shape and properties.
        
        Args:
            input_tensor: Input tensor to validate
            
        Raises:
            ValueError: If input tensor is invalid
        """
        if input_tensor.dim() != 3:
            raise ValueError(f"Input tensor must be 3-dimensional [batch, variables, sequence], got {input_tensor.dim()}D tensor")
        
        batch, variables, sequence = input_tensor.shape
        
        if variables != self.config.num_variables:
            raise ValueError(f"Input variables count {variables} does not match configured count {self.config.num_variables}")
        
        if sequence != self.config.sequence_length:
            raise ValueError(f"Input sequence length {sequence} does not match configured length {self.config.sequence_length}")
        
        # Check for invalid values
        if torch.isnan(input_tensor).any():
            # Note: We allow NaN to propagate through the model as per test requirements
            pass
        
        if torch.isinf(input_tensor).any():
            # Note: We allow Inf to propagate through the model as per test requirements
            pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model architecture and dimensions.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'config': {
                'sequence_length': self.config.sequence_length,
                'num_variables': self.config.num_variables,
                'patch_size': self.config.patch_size,
                'num_encoder_layers': self.config.num_encoder_layers,
            },
            'dimensions': {
                'input_shape': f"[batch, {self.config.num_variables}, {self.config.sequence_length}]",
                'encoder_input_shape': f"[batch, {self.psformer_dims['N']}, {self.psformer_dims['C']}]",
                'num_patches': self.psformer_dims['N'],
                'segment_length': self.psformer_dims['C'],
            },
            'components': {
                'revin_features': self.revin_layer.num_features,
                'data_transformer_config': {
                    'patch_size': self.data_transformer.config.patch_size,
                    'sequence_length': self.data_transformer.config.sequence_length,
                    'num_variables': self.data_transformer.config.num_variables,
                    'num_patches': self.data_transformer.config.num_patches,
                    'segment_length': self.data_transformer.config.segment_length,
                },
                'encoder_layers': len(self.encoder.layers),
                'ps_block_dimension': self.encoder.layers[0].ps_block.N if self.encoder.layers else None,
            }
        }
    



def create_psformer_model(sequence_length: int, 
                         num_variables: int, 
                         patch_size: int, 
                         num_encoder_layers: int,
                         prediction_length: int,
                         **kwargs) -> PSformer:
    """
    Factory function to create a PSformer model with default configuration.
    
    Args:
        sequence_length: Total input sequence length (L)
        num_variables: Number of time series variables (M)
        patch_size: Size of each temporal patch (P)
        num_encoder_layers: Number of PSformer encoder layers
        prediction_length: Length of prediction horizon (F)
        **kwargs: Additional configuration parameters
        
    Returns:
        PSformer model instance
    """
    config = PSformerConfig(
        sequence_length=sequence_length,
        num_variables=num_variables,
        patch_size=patch_size,
        num_encoder_layers=num_encoder_layers,
        prediction_length=prediction_length,
        **kwargs
    )
    return PSformer(config)
