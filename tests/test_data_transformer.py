import pytest
import torch
import numpy as np
from src.blocks.data_transformer import (
    DataTransformationConfig, 
    PSformerDataTransformer, 
    create_transformer_for_psformer,
    get_psformer_dimensions
)


class TestDataTransformationConfig:
    """Test cases for DataTransformationConfig class"""
    
    def test_config_creation_valid(self):
        """Test valid configuration creation"""
        config = DataTransformationConfig(patch_size=16, sequence_length=96, num_variables=7)
        assert config.patch_size == 16
        assert config.sequence_length == 96
        assert config.num_variables == 7
        assert config.num_patches == 6  # 96 / 16
        assert config.segment_length == 112  # 7 * 16
    
    def test_config_validation_errors(self):
        """Test configuration validation errors"""
        # Sequence length not divisible by patch size
        with pytest.raises(ValueError):
            DataTransformationConfig(patch_size=17, sequence_length=96, num_variables=7)
        
        # Invalid patch size
        with pytest.raises(ValueError):
            DataTransformationConfig(patch_size=0, sequence_length=96, num_variables=7)
            
        with pytest.raises(ValueError):
            DataTransformationConfig(patch_size=-1, sequence_length=96, num_variables=7)
        
        # Invalid number of variables
        with pytest.raises(ValueError):
            DataTransformationConfig(patch_size=16, sequence_length=96, num_variables=0)
            
        with pytest.raises(ValueError):
            DataTransformationConfig(patch_size=16, sequence_length=96, num_variables=-1)


class TestDataTransformationAtomic:
    """Atomic transformation tests"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.config = DataTransformationConfig(patch_size=16, sequence_length=96, num_variables=7)
        self.transformer = PSformerDataTransformer(self.config)
    
    def test_create_patches_basic(self):
        """Test basic patch creation"""
        # Test: [batch=2, M=7, L=96] -> [batch=2, M=7, N=6, P=16]
        input_tensor = torch.randn(2, 7, 96)
        patches = self.transformer.create_patches(input_tensor)
        
        assert patches.shape == (2, 7, 6, 16)
        # Check that patch content is correct
        assert torch.allclose(patches[0, 0, 0, :], input_tensor[0, 0, 0:16])
        assert torch.allclose(patches[0, 0, 1, :], input_tensor[0, 0, 16:32])
    
    def test_create_patches_validation(self):
        """Test patch creation validation"""
        # Invalid input dimensions
        with pytest.raises(ValueError):
            invalid_input = torch.randn(2, 7)  # 2D tensor
            self.transformer.create_patches(invalid_input)
        
        # Wrong sequence length
        with pytest.raises(ValueError):
            invalid_input = torch.randn(2, 7, 100)  # Wrong sequence length
            self.transformer.create_patches(invalid_input)
        
        # Wrong number of variables
        with pytest.raises(ValueError):
            invalid_input = torch.randn(2, 8, 96)  # Wrong variable count
            self.transformer.create_patches(invalid_input)
    
    def test_create_segments_basic(self):
        """Test basic segment creation"""
        # Test: [batch=2, M=7, N=6, P=16] -> [batch=2, N=6, C=112]
        patches = torch.randn(2, 7, 6, 16)
        segments = self.transformer.create_segments(patches)
        
        assert segments.shape == (2, 6, 112)  # C = 7*16 = 112
        # Check that content from different variables is properly concatenated
        assert torch.allclose(segments[0, 0, 0:16], patches[0, 0, 0, :])  # Variable 0, patch 0
        assert torch.allclose(segments[0, 0, 16:32], patches[0, 1, 0, :])  # Variable 1, patch 0
    
    def test_create_segments_validation(self):
        """Test segment creation validation"""
        # Invalid input dimensions
        with pytest.raises(ValueError):
            invalid_input = torch.randn(2, 7, 96)  # 3D tensor instead of 4D
            self.transformer.create_segments(invalid_input)
        
        # Wrong variable count
        with pytest.raises(ValueError):
            invalid_input = torch.randn(2, 8, 6, 16)  # Wrong variable count
            self.transformer.create_segments(invalid_input)
        
        # Wrong patch count
        with pytest.raises(ValueError):
            invalid_input = torch.randn(2, 7, 5, 16)  # Wrong patch count
            self.transformer.create_segments(invalid_input)
        
        # Wrong patch size
        with pytest.raises(ValueError):
            invalid_input = torch.randn(2, 7, 6, 17)  # Wrong patch size
            self.transformer.create_segments(invalid_input)
    
    def test_restore_shape_basic(self):
        """Test basic shape restoration"""
        # Test: [batch=2, N=6, C=112] -> [batch=2, M=7, L=96]
        segments = torch.randn(2, 6, 112)
        restored = self.transformer.restore_shape(segments, target_sequence_length=96)
        
        assert restored.shape == (2, 7, 96)
    
    def test_restore_shape_validation(self):
        """Test shape restoration validation"""
        # Invalid input dimensions
        with pytest.raises(ValueError):
            invalid_input = torch.randn(2, 7, 6, 16)  # 4D tensor instead of 3D
            self.transformer.restore_shape(invalid_input, target_sequence_length=96)
        
        # Wrong segment length
        with pytest.raises(ValueError):
            invalid_input = torch.randn(2, 6, 113)  # Wrong segment length
            self.transformer.restore_shape(invalid_input, target_sequence_length=96)
        
        # Wrong patch count
        with pytest.raises(ValueError):
            invalid_input = torch.randn(2, 5, 112)  # Wrong patch count
            self.transformer.restore_shape(invalid_input, target_sequence_length=96)
        
        # Incompatible target sequence length
        with pytest.raises(ValueError):
            invalid_input = torch.randn(2, 6, 112)
            self.transformer.restore_shape(invalid_input, target_sequence_length=100)  # Incompatible length


class TestDataTransformationIntegration:
    """Integration tests for the data transformation pipeline"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.config = DataTransformationConfig(patch_size=16, sequence_length=512, num_variables=7)
        self.transformer = PSformerDataTransformer(self.config)
    
    def test_psformer_encoder_input_compatibility(self):
        """Test that transformed data works with PSformer Encoder"""
        # Test that transformed data has the correct shape for PSformer
        input_data = torch.randn(16, 7, 512)
        segments = self.transformer.forward_transform(input_data)
        
        # segments shape should be [16, 32, 112] where C=7*16=112
        assert segments.shape == (16, 32, 112)
    
    def test_full_psformer_pipeline(self):
        """Test complete data flow: Input -> Transform -> Restore"""
        original_input = torch.randn(16, 7, 512)
        
        # Forward through PSformer
        segments = self.transformer.forward_transform(original_input)
        
        # Simulate PSformer processing (just pass through for this test)
        processed_segments = segments  # Mock processing
        
        # Transform back for output
        restored = self.transformer.inverse_transform(processed_segments, 512)
        
        assert restored.shape == original_input.shape
        # Shapes should be compatible throughout pipeline
        assert segments.shape == (16, 32, 112)
        assert processed_segments.shape == (16, 32, 112)
        assert restored.shape == (16, 7, 512)
    
    def test_revin_integration(self):
        """Test integration with RevIN normalization"""
        from src.blocks.RevIN import RevIN
        
        input_data = torch.randn(16, 7, 512)
        revin = RevIN(num_features=7)
        transformer = PSformerDataTransformer(
            DataTransformationConfig(patch_size=16, sequence_length=512, num_variables=7)
        )
        
        # Normalize with RevIN
        normalized = revin(input_data, mode='norm')
        
        # Transform to segments
        segments = transformer.forward_transform(normalized)
        
        # Restore shape
        restored_segments = transformer.inverse_transform(segments, 512)
        
        # Denormalize with RevIN
        denormalized = revin(restored_segments, mode='denorm')
        
        # Should be close to original input
        assert torch.allclose(denormalized, input_data, atol=1e-5)


class TestDataTransformationProperties:
    """Property-based tests for data transformation"""
    
    def test_transformation_symmetry(self):
        """Test that forward + inverse = identity"""
        # Test with different configurations
        configs = [
            DataTransformationConfig(patch_size=16, sequence_length=96, num_variables=7),
            DataTransformationConfig(patch_size=8, sequence_length=96, num_variables=21),
            DataTransformationConfig(patch_size=32, sequence_length=96, num_variables=1)
        ]
        
        test_shapes = [
            (16, 7, 96),   # Matches first config
            (8, 21, 96),   # Matches second config
            (32, 1, 96)    # Matches third config
        ]
        
        for config, shape in zip(configs, test_shapes):
            transformer = PSformerDataTransformer(config)
            input_tensor = torch.randn(*shape)
            segments = transformer.forward_transform(input_tensor)
            restored = transformer.inverse_transform(segments, input_tensor.shape[-1])
            
            assert torch.allclose(input_tensor, restored, atol=1e-6)
    
    def test_element_preservation(self):
        """Test that no elements are lost/gained in transformation"""
        config = DataTransformationConfig(patch_size=16, sequence_length=512, num_variables=7)
        transformer = PSformerDataTransformer(config)
        
        input_tensor = torch.randn(16, 7, 512)
        segments = transformer.forward_transform(input_tensor)
        
        # Sum should be approximately equal (with floating point precision)
        assert torch.allclose(torch.sum(input_tensor), torch.sum(segments), atol=1e-4)
        
        # Number of elements should be exactly equal
        assert torch.numel(input_tensor) == torch.numel(segments)


class TestDimensionCompatibility:
    """Dimension compatibility tests"""
    
    @pytest.mark.parametrize("dataset_config", [
        {"num_variables": 7, "sequence_length": 512, "patch_size": 16},   # ETTh1-like
        {"num_variables": 7, "sequence_length": 512, "patch_size": 16},   # ETTm1-like
        {"num_variables": 21, "sequence_length": 512, "patch_size": 16},  # Weather-like
        {"num_variables": 862, "sequence_length": 512, "patch_size": 16}, # Traffic-like
    ])
    def test_psformer_encoder_dimensions(self, dataset_config):
        """Test dimensions match PSformer paper specifications"""
        M = dataset_config["num_variables"]
        L = dataset_config["sequence_length"]
        P = dataset_config["patch_size"]
        
        transformer = PSformerDataTransformer(
            DataTransformationConfig(patch_size=P, sequence_length=L, num_variables=M)
        )
        
        # Create dummy input
        input_tensor = torch.randn(1, M, L)
        segments = transformer.forward_transform(input_tensor)
        
        N = L // P  # Number of segments
        C = M * P   # Segment length
        
        assert segments.shape == (1, N, C)
    
    def test_attention_mechanism_compatibility(self):
        """Test that segments work with scaled dot-product attention"""
        config = DataTransformationConfig(patch_size=16, sequence_length=512, num_variables=7)
        transformer = PSformerDataTransformer(config)
        
        # Create segments
        segments = torch.randn(16, 32, 112)  # batch=16, N=32, C=112
        
        # Use as Q, K, V matrices for attention
        Q = K = V = segments
        
        # Simple attention computation (scaled dot-product)
        scale = torch.sqrt(torch.tensor(112, dtype=torch.float32))
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # Verify output shape
        assert attention_output.shape == (16, 32, 112)


class TestDataTransformationErrors:
    """Error handling tests"""
    
    def test_invalid_dimensions(self):
        """Test handling of invalid dimensions"""
        # L not divisible by P
        with pytest.raises(ValueError):
            transformer = PSformerDataTransformer(
                DataTransformationConfig(patch_size=17, sequence_length=512, num_variables=7)
            )
            input_tensor = torch.randn(16, 7, 512)
            transformer.forward_transform(input_tensor)
    
    def test_mismatched_restore_dimensions(self):
        """Test handling of mismatched restore dimensions"""
        transformer = PSformerDataTransformer(
            DataTransformationConfig(patch_size=16, sequence_length=512, num_variables=7)
        )
        
        # Wrong target length
        segments = torch.randn(16, 32, 112)
        with pytest.raises(ValueError):
            transformer.inverse_transform(segments, target_sequence_length=100)  # Wrong length


class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_create_transformer_for_psformer(self):
        """Test factory method for PSformer"""
        transformer = create_transformer_for_psformer(
            sequence_length=512,
            num_variables=7,
            patch_size=16
        )
        
        assert isinstance(transformer, PSformerDataTransformer)
        assert transformer.config.sequence_length == 512
        assert transformer.config.num_variables == 7
        assert transformer.config.patch_size == 16
        assert transformer.config.num_patches == 32
        assert transformer.config.segment_length == 112
    
    def test_get_psformer_dimensions(self):
        """Test getting PSformer dimensions"""
        transformer = create_transformer_for_psformer(
            sequence_length=512,
            num_variables=7,
            patch_size=16
        )
        
        dims = get_psformer_dimensions(transformer)
        
        assert dims["C"] == 112  # 7 * 16
        assert dims["N"] == 32   # 512 / 16
        assert dims["segment_shape"] == (None, 32, 112)


class TestAdditionalProperties:
    """Additional property tests"""
    
    def test_gradient_flow(self):
        """Test that gradients flow through transformations"""
        config = DataTransformationConfig(patch_size=16, sequence_length=96, num_variables=7)
        transformer = PSformerDataTransformer(config)
        
        # Input with gradients
        input_tensor = torch.randn(2, 7, 96, requires_grad=True)
        
        # Forward pass
        segments = transformer.forward_transform(input_tensor)
        loss = segments.sum()
        
        # Backward pass
        loss.backward()
        
        # Gradients should exist
        assert input_tensor.grad is not None
        assert not torch.all(input_tensor.grad == 0)
    
    def test_device_placement_consistency(self):
        """Test device placement consistency"""
        config = DataTransformationConfig(patch_size=16, sequence_length=96, num_variables=7)
        transformer = PSformerDataTransformer(config)
        
        # Test on CPU
        input_cpu = torch.randn(2, 7, 96)
        segments_cpu = transformer.forward_transform(input_cpu)
        assert segments_cpu.device == input_cpu.device
        
        # Test on GPU if available
        if torch.cuda.is_available():
            input_cuda = input_cpu.cuda()
            transformer_cuda = PSformerDataTransformer(config)
            segments_cuda = transformer_cuda.forward_transform(input_cuda)
            assert segments_cuda.device == input_cuda.device