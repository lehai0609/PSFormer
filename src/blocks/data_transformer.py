import torch
import torch.nn as nn


class DataTransformationConfig:
    """
    Configuration class for data transformation parameters
    """
    def __init__(self, patch_size: int, sequence_length: int, num_variables: int):
        """
        Args:
            patch_size: Size of each temporal patch (P)
            sequence_length: Total input sequence length (L)
            num_variables: Number of time series variables (M)
        """
        self.patch_size = patch_size
        self.sequence_length = sequence_length
        self.num_variables = num_variables
        
        # Validate and calculate derived parameters
        self._validate()
        self.num_patches = self.sequence_length // self.patch_size
        self.segment_length = self.num_variables * self.patch_size
    
    def _validate(self):
        """Validate configuration parameters"""
        if self.patch_size <= 0:
            raise ValueError(f"Patch size must be positive, got {self.patch_size}")
        if self.sequence_length % self.patch_size != 0:
            raise ValueError(f"Sequence length {self.sequence_length} must be divisible by patch size {self.patch_size}")
        if self.num_variables <= 0:
            raise ValueError(f"Number of variables must be positive, got {self.num_variables}")


class PSformerDataTransformer:
    """
    Data transformation utility for PSformer that handles conversion between
    standard time series format and PSformer's segment-based representation.
    """
    
    def __init__(self, config: DataTransformationConfig):
        """
        Args:
            config: DataTransformationConfig instance with transformation parameters
        """
        self.config = config
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate the configuration"""
        if not isinstance(self.config, DataTransformationConfig):
            raise TypeError("config must be an instance of DataTransformationConfig")
    
    def create_patches(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Transform: [batch, variables, sequence] -> [batch, variables, num_patches, patch_size]
        
        Args:
            input_tensor: Input tensor of shape [batch, variables, sequence_length]
            
        Returns:
            Patched tensor of shape [batch, variables, num_patches, patch_size]
        """
        # Input validation
        if input_tensor.dim() != 3:
            raise ValueError(f"Input tensor must be 3-dimensional, got {input_tensor.dim()}")
            
        batch, variables, sequence = input_tensor.shape
        
        if sequence != self.config.sequence_length:
            raise ValueError(f"Input sequence length {sequence} does not match configured length {self.config.sequence_length}")
            
        if variables != self.config.num_variables:
            raise ValueError(f"Input variables count {variables} does not match configured count {self.config.num_variables}")
        
        # Reshape sequence dimension to split into patches
        # [batch, variables, sequence] -> [batch, variables, num_patches, patch_size]
        patched = input_tensor.view(batch, variables, self.config.num_patches, self.config.patch_size)
        
        return patched
    
    def create_segments(self, patched_tensor: torch.Tensor) -> torch.Tensor:
        """
        Transform: [batch, variables, num_patches, patch_size] -> [batch, num_patches, segment_length]
        
        Args:
            patched_tensor: Patched tensor of shape [batch, variables, num_patches, patch_size]
            
        Returns:
            Segmented tensor of shape [batch, num_patches, segment_length]
        """
        # Input validation
        if patched_tensor.dim() != 4:
            raise ValueError(f"Patched tensor must be 4-dimensional, got {patched_tensor.dim()}")
            
        batch, variables, num_patches, patch_size = patched_tensor.shape
        
        if variables != self.config.num_variables:
            raise ValueError(f"Patched tensor variables count {variables} does not match configured count {self.config.num_variables}")
            
        if num_patches != self.config.num_patches:
            raise ValueError(f"Patched tensor num_patches {num_patches} does not match configured count {self.config.num_patches}")
            
        if patch_size != self.config.patch_size:
            raise ValueError(f"Patched tensor patch_size {patch_size} does not match configured size {self.config.patch_size}")
        
        # Step 1: Transpose to put patches before variables
        # [batch, variables, num_patches, patch_size] -> [batch, num_patches, variables, patch_size]
        transposed = patched_tensor.transpose(1, 2)
        
        # Step 2: Reshape to create segments by concatenating all variables for each patch
        # [batch, num_patches, variables, patch_size] -> [batch, num_patches, variables*patch_size]
        segments = transposed.contiguous().view(batch, num_patches, self.config.segment_length)
        
        return segments
    
    def restore_shape(self, segment_tensor: torch.Tensor, target_sequence_length: int) -> torch.Tensor:
        """
        Transform: [batch, num_patches, segment_length] -> [batch, variables, sequence]
        
        Args:
            segment_tensor: Segmented tensor of shape [batch, num_patches, segment_length]
            target_sequence_length: Target sequence length for output
            
        Returns:
            Restored tensor of shape [batch, variables, target_sequence_length]
        """
        # Input validation
        if segment_tensor.dim() != 3:
            raise ValueError(f"Segment tensor must be 3-dimensional, got {segment_tensor.dim()}")
            
        batch, num_patches, segment_length = segment_tensor.shape
        
        if segment_length != self.config.segment_length:
            raise ValueError(f"Segment tensor segment_length {segment_length} does not match configured length {self.config.segment_length}")
            
        if num_patches != self.config.num_patches:
            raise ValueError(f"Segment tensor num_patches {num_patches} does not match configured count {self.config.num_patches}")
            
        if num_patches * self.config.patch_size != target_sequence_length:
            raise ValueError(f"Target sequence length {target_sequence_length} is incompatible with num_patches {num_patches} and patch_size {self.config.patch_size}")
        
        # Step 1: Reshape segments back to separate variables and patches
        # [batch, num_patches, segment_length] -> [batch, num_patches, variables, patch_size]
        reshaped = segment_tensor.view(batch, num_patches, self.config.num_variables, self.config.patch_size)
        
        # Step 2: Transpose to put variables before patches
        # [batch, num_patches, variables, patch_size] -> [batch, variables, num_patches, patch_size]
        transposed = reshaped.transpose(1, 2)
        
        # Step 3: Reshape to flatten patches back into sequence
        # [batch, variables, num_patches, patch_size] -> [batch, variables, num_patches*patch_size]
        output = transposed.contiguous().view(batch, self.config.num_variables, target_sequence_length)
        
        return output
    
    def forward_transform(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Complete pipeline: input -> patches -> segments
        
        Args:
            input_tensor: Input tensor of shape [batch, variables, sequence_length]
            
        Returns:
            Segmented tensor of shape [batch, num_patches, segment_length]
        """
        patches = self.create_patches(input_tensor)
        segments = self.create_segments(patches)
        return segments
    
    def inverse_transform(self, segment_tensor: torch.Tensor, target_sequence_length: int) -> torch.Tensor:
        """
        Complete pipeline: segments -> patches -> output
        
        Args:
            segment_tensor: Segmented tensor of shape [batch, num_patches, segment_length]
            target_sequence_length: Target sequence length for output
            
        Returns:
            Restored tensor of shape [batch, variables, target_sequence_length]
        """
        return self.restore_shape(segment_tensor, target_sequence_length)
    
    def calculate_output_dimensions(self, input_shape: tuple) -> dict:
        """
        Given input shape, calculate all intermediate shapes
        
        Args:
            input_shape: Input shape tuple (batch, variables, sequence)
            
        Returns:
            Dictionary with intermediate shapes
        """
        batch, variables, sequence = input_shape
        
        patch_shape = (batch, variables, self.config.num_patches, self.config.patch_size)
        segment_shape = (batch, self.config.num_patches, self.config.segment_length)
        
        return {
            'patches': patch_shape,
            'segments': segment_shape,
            'segment_length': self.config.segment_length,
            'num_patches': self.config.num_patches
        }
    
    def verify_transformation_symmetry(self, input_tensor: torch.Tensor) -> bool:
        """
        Test that forward + inverse = identity (within numerical precision)
        
        Args:
            input_tensor: Input tensor to test
            
        Returns:
            True if transformation is symmetric
        """
        segments = self.forward_transform(input_tensor)
        restored = self.inverse_transform(segments, input_tensor.shape[-1])
        return torch.allclose(input_tensor, restored, atol=1e-6)


def create_transformer_for_psformer(sequence_length: int, num_variables: int, patch_size: int) -> PSformerDataTransformer:
    """
    Factory method with PSformer-specific defaults
    
    Args:
        sequence_length: Total input sequence length (L)
        num_variables: Number of time series variables (M)
        patch_size: Size of each temporal patch (P)
        
    Returns:
        PSformerDataTransformer instance
    """
    config = DataTransformationConfig(
        patch_size=patch_size,
        sequence_length=sequence_length,
        num_variables=num_variables
    )
    return PSformerDataTransformer(config)


def get_psformer_dimensions(transformer: PSformerDataTransformer) -> dict:
    """
    Return the key dimensions that PSformer encoder needs
    
    Args:
        transformer: PSformerDataTransformer instance
        
    Returns:
        Dictionary with key dimensions
    """
    return {
        'C': transformer.config.segment_length,      # This becomes the feature dimension for attention
        'N': transformer.config.num_patches,         # This becomes the sequence dimension for attention
        'segment_shape': (None, transformer.config.num_patches, transformer.config.segment_length)  # Final shape for PSformer input
    }