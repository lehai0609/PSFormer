import pytest
import torch
import numpy as np
from src.blocks import PSformer, PSformerConfig, create_psformer_model


class TestInputProcessingPipeline:
    """
    Comprehensive test suite for PSformer input processing pipeline.
    Tests the complete flow: Raw Input → RevIN → Data Transformation → Encoder
    """
    
    @pytest.fixture
    def default_config(self):
        """Default configuration for tests"""
        return {
            'sequence_length': 512,
            'patch_size': 32,
            'num_variables': 7,
            'num_encoder_layers': 3,
            'prediction_length': 96
        }
    
    @pytest.fixture
    def psformer_model(self, default_config):
        """Create PSformer model with default configuration"""
        return create_psformer_model(**default_config)
    
    @pytest.fixture
    def derived_values(self, default_config):
        """Calculate derived values for assertions"""
        return {
            'batch_size': 4,
            'N_segments': default_config['sequence_length'] // default_config['patch_size'],  # 16
            'C_segment_length': default_config['num_variables'] * default_config['patch_size']  # 224
        }
    
    @pytest.fixture
    def valid_input(self, default_config, derived_values):
        """Create valid input tensor for testing"""
        return torch.randn(
            derived_values['batch_size'], 
            default_config['num_variables'], 
            default_config['sequence_length']
        )

    # ===== 1. DATA VALIDATION TESTS =====
    
    def test_pipeline_rejects_input_with_incorrect_dimensions(self, psformer_model, derived_values, default_config):
        """Test that pipeline fails with incorrect input dimensions"""
        batch_size = derived_values['batch_size']
        
        # Test 2D input
        input_2d = torch.randn(batch_size, default_config['sequence_length'])
        with pytest.raises(ValueError, match="must be 3-dimensional"):
            psformer_model(input_2d)
        
        # Test 4D input
        input_4d = torch.randn(batch_size, default_config['num_variables'], default_config['sequence_length'], 1)
        with pytest.raises(ValueError, match="must be 3-dimensional"):
            psformer_model(input_4d)
    
    def test_pipeline_rejects_mismatched_sequence_length(self, psformer_model, derived_values, default_config):
        """Test that pipeline fails with wrong sequence length"""
        batch_size = derived_values['batch_size']
        
        # Wrong sequence length (not divisible by patch_size or doesn't match config)
        wrong_length_tensor = torch.randn(batch_size, default_config['num_variables'], 100)
        
        with pytest.raises(ValueError, match="Input sequence length .* does not match configured length"):
            psformer_model(wrong_length_tensor)
    
    def test_pipeline_rejects_mismatched_num_variables(self, psformer_model, derived_values, default_config):
        """Test that pipeline fails with wrong number of variables"""
        batch_size = derived_values['batch_size']
        
        # Wrong number of variables
        wrong_variables_tensor = torch.randn(batch_size, 5, default_config['sequence_length'])
        
        with pytest.raises(ValueError, match="Input variables count .* does not match configured count"):
            psformer_model(wrong_variables_tensor)

    # ===== 2. PROCESSING/FEATURE ENGINEERING TESTS =====
    
    def test_revin_normalizes_data_per_channel(self, psformer_model, derived_values, default_config):
        """Test that RevIN normalizes each channel independently"""
        batch_size = derived_values['batch_size']
        
        # Create input where each channel has distinct statistics
        raw_input = torch.randn(batch_size, default_config['num_variables'], default_config['sequence_length'])
        
        # Modify specific channels to have different means and variances
        raw_input[:, 0, :] = raw_input[:, 0, :] + 10  # Shift mean of first channel
        raw_input[:, 1, :] = raw_input[:, 1, :] * 5   # Scale variance of second channel
        
        # Store original statistics
        original_mean_ch0 = torch.mean(raw_input[:, 0, :], dim=-1, keepdim=True)
        original_mean_ch1 = torch.mean(raw_input[:, 1, :], dim=-1, keepdim=True)
        original_std_ch0 = torch.std(raw_input[:, 0, :], dim=-1, keepdim=True)
        original_std_ch1 = torch.std(raw_input[:, 1, :], dim=-1, keepdim=True)
        
        # Apply normalization through the model
        normalized_input = psformer_model.revin_layer(raw_input, mode='norm')
        
        # Check that each channel is normalized independently
        normalized_mean_ch0 = torch.mean(normalized_input[:, 0, :], dim=-1)
        normalized_std_ch0 = torch.std(normalized_input[:, 0, :], dim=-1)
        normalized_mean_ch1 = torch.mean(normalized_input[:, 1, :], dim=-1)
        normalized_std_ch1 = torch.std(normalized_input[:, 1, :], dim=-1)
        
        # Assert normalization worked (use more realistic tolerances for numerical precision)
        assert torch.allclose(normalized_mean_ch0, torch.zeros_like(normalized_mean_ch0), atol=1e-5)
        assert torch.allclose(normalized_std_ch0, torch.ones_like(normalized_std_ch0), atol=1e-2)
        assert torch.allclose(normalized_mean_ch1, torch.zeros_like(normalized_mean_ch1), atol=1e-5)
        assert torch.allclose(normalized_std_ch1, torch.ones_like(normalized_std_ch1), atol=1e-2)
    
    def test_revin_stores_statistics_for_denormalization(self, psformer_model, valid_input):
        """Test that RevIN stores statistics correctly"""
        # Process input through the model
        psformer_model(valid_input)
        
        # Check that statistics are stored
        assert hasattr(psformer_model.revin_layer, 'mean')
        assert hasattr(psformer_model.revin_layer, 'stdev')
        assert psformer_model.revin_layer.mean is not None
        assert psformer_model.revin_layer.stdev is not None
        
        # Check statistics shape - should be ready for broadcasting
        # Shape should be [batch, channels, 1] for per-channel statistics
        expected_shape = (valid_input.shape[0], valid_input.shape[1], 1)
        assert psformer_model.revin_layer.mean.shape == expected_shape
        assert psformer_model.revin_layer.stdev.shape == expected_shape

    # ===== 3. MODEL BEHAVIOR TESTS =====
    
    def test_pipeline_output_shape_is_correct(self, psformer_model, valid_input, derived_values):
        """Test that the complete pipeline produces correct output shape"""
        encoder_output, attention_weights = psformer_model(valid_input)
        
        # Check encoder output shape: [batch, N_segments, C_segment_length]
        expected_shape = (derived_values['batch_size'], derived_values['N_segments'], derived_values['C_segment_length'])
        assert encoder_output.shape == expected_shape
    
    def test_attention_weights_have_correct_shape_and_structure(self, psformer_model, valid_input, derived_values, default_config):
        """Test that attention weights have correct shape and structure"""
        encoder_output, attention_weights_list = psformer_model(valid_input)
        
        # Check that we have attention weights for each encoder layer
        assert len(attention_weights_list) == default_config['num_encoder_layers']
        
        # Check each layer's attention weights
        for layer_weights in attention_weights_list:
            # Each layer should have tuple of (stage1_weights, stage2_weights)
            assert isinstance(layer_weights, tuple)
            assert len(layer_weights) == 2
            
            stage1_weights, stage2_weights = layer_weights
            
            # Attention is over dimension C, so weights should be [batch, C, C]
            expected_attention_shape = (derived_values['batch_size'], derived_values['C_segment_length'], derived_values['C_segment_length'])
            assert stage1_weights.shape == expected_attention_shape
            assert stage2_weights.shape == expected_attention_shape
            
            # Attention weights should sum to 1 along the last dimension
            assert torch.allclose(torch.sum(stage1_weights, dim=-1), torch.ones_like(torch.sum(stage1_weights, dim=-1)), atol=1e-5)
            assert torch.allclose(torch.sum(stage2_weights, dim=-1), torch.ones_like(torch.sum(stage2_weights, dim=-1)), atol=1e-5)

    # ===== 4. PERFORMANCE AND ROBUSTNESS TESTS =====
    
    def test_pipeline_handles_input_with_nan(self, psformer_model, valid_input):
        """Test that pipeline doesn't crash with NaN input"""
        # Create input with NaN
        input_with_nan = valid_input.clone()
        input_with_nan[0, 0, 0] = float('nan')
        
        # Model should not crash, NaN should propagate
        encoder_output, _ = psformer_model(input_with_nan)
        
        # Check that NaN propagated (output should contain NaN)
        assert torch.isnan(encoder_output).any()
    
    def test_pipeline_handles_input_with_inf(self, psformer_model, valid_input):
        """Test that pipeline doesn't crash with Inf input"""
        # Create input with Inf
        input_with_inf = valid_input.clone()
        input_with_inf[0, 0, 0] = float('inf')
        
        # Model should not crash, Inf should propagate
        encoder_output, _ = psformer_model(input_with_inf)
        
        # Check that Inf propagated (output should contain Inf or NaN from calculations)
        assert torch.isinf(encoder_output).any() or torch.isnan(encoder_output).any()
    
    def test_pipeline_handles_zero_variance_channel(self, psformer_model, derived_values, default_config):
        """Test that pipeline handles channels with zero variance"""
        batch_size = derived_values['batch_size']
        
        # Create input with zero variance (constant values)
        input_zero_var = torch.ones(batch_size, default_config['num_variables'], default_config['sequence_length'])
        
        # Model should not crash due to division by zero in normalization
        encoder_output, _ = psformer_model(input_zero_var)
        
        # Output should be finite (no NaN or Inf from division by zero)
        assert torch.isfinite(encoder_output).all()

    # ===== 5. ARCHITECTURE TESTS =====
    
    def test_encoder_ps_block_dimension_matches_data_transformer_segment_length(self, psformer_model, derived_values):
        """Test critical architecture constraint: PS Block dimension matches segment length"""
        # Get segment length from data transformer
        C_from_transformer = psformer_model.data_transformer.config.segment_length
        
        # Get PS Block dimension from first encoder layer
        N_from_ps_block = psformer_model.encoder.layers[0].ps_block.N
        
        # These must match for the architecture to work correctly
        assert C_from_transformer == N_from_ps_block
        assert N_from_ps_block == derived_values['C_segment_length']
    
    def test_data_transformer_configuration_consistency(self, psformer_model, default_config, derived_values):
        """Test that data transformer configuration is consistent with model config"""
        transformer_config = psformer_model.data_transformer.config
        
        assert transformer_config.sequence_length == default_config['sequence_length']
        assert transformer_config.num_variables == default_config['num_variables']
        assert transformer_config.patch_size == default_config['patch_size']
        assert transformer_config.num_patches == derived_values['N_segments']
        assert transformer_config.segment_length == derived_values['C_segment_length']
    
    def test_revin_configuration_consistency(self, psformer_model, default_config):
        """Test that RevIN configuration is consistent with model config"""
        assert psformer_model.revin_layer.num_features == default_config['num_variables']
    
    def test_model_info_contains_correct_information(self, psformer_model, default_config, derived_values):
        """Test that model info method returns correct architecture information"""
        info = psformer_model.get_model_info()
        
        # Check config information
        assert info['config']['sequence_length'] == default_config['sequence_length']
        assert info['config']['num_variables'] == default_config['num_variables']
        assert info['config']['patch_size'] == default_config['patch_size']
        assert info['config']['num_encoder_layers'] == default_config['num_encoder_layers']
        
        # Check dimensions
        assert info['dimensions']['num_patches'] == derived_values['N_segments']
        assert info['dimensions']['segment_length'] == derived_values['C_segment_length']
        
        # Check components
        assert info['components']['revin_features'] == default_config['num_variables']
        assert info['components']['encoder_layers'] == default_config['num_encoder_layers']
        assert info['components']['ps_block_dimension'] == derived_values['C_segment_length']

    # ===== 6. INTEGRATION TESTS =====
    
    def test_forward_and_inverse_transformation_symmetry(self, psformer_model, valid_input):
        """Test that forward transformation can be inverted"""
        # Test the data transformer symmetry
        segments = psformer_model.data_transformer.forward_transform(valid_input)
        restored = psformer_model.data_transformer.inverse_transform(segments, valid_input.shape[-1])
        
        # Should be able to restore original input (within numerical precision)
        assert torch.allclose(valid_input, restored, atol=1e-6)
    
    def test_revin_normalization_and_denormalization_symmetry(self, psformer_model, valid_input):
        """Test that RevIN normalization can be reversed"""
        # Normalize
        normalized = psformer_model.revin_layer(valid_input, mode='norm')
        
        # Denormalize
        denormalized = psformer_model.revin_layer(normalized, mode='denorm')
        
        # Should restore original input (within numerical precision)
        assert torch.allclose(valid_input, denormalized, atol=1e-5)
    
    def test_complete_pipeline_with_different_batch_sizes(self, default_config):
        """Test pipeline works with different batch sizes"""
        model = create_psformer_model(**default_config)
        
        for batch_size in [1, 2, 8, 16]:
            test_input = torch.randn(batch_size, default_config['num_variables'], default_config['sequence_length'])
            
            # Should not crash and should produce correct output shape
            final_predictions = model(test_input)
            
            expected_shape = (batch_size, default_config['num_variables'], default_config['prediction_length'])
            
            assert final_predictions.shape == expected_shape

    # ===== 7. EDGE CASE TESTS =====
    
    def test_pipeline_with_minimal_configuration(self):
        """Test pipeline with minimal valid configuration"""
        minimal_config = {
            'sequence_length': 4,  # Minimal sequence that can be patched
            'patch_size': 2,
            'num_variables': 1,
            'num_encoder_layers': 1,
            'prediction_length': 2
        }
        
        model = create_psformer_model(**minimal_config)
        test_input = torch.randn(1, 1, 4)
        
        final_predictions = model(test_input)
        
        # Should work correctly even with minimal configuration
        expected_shape = (1, 1, 2)  # batch=1, variables=1, prediction_length=2
        assert final_predictions.shape == expected_shape
    
    def test_pipeline_with_large_configuration(self):
        """Test pipeline with larger configuration"""
        large_config = {
            'sequence_length': 1024,
            'patch_size': 64,
            'num_variables': 12,
            'num_encoder_layers': 6,
            'prediction_length': 192
        }
        
        model = create_psformer_model(**large_config)
        test_input = torch.randn(2, 12, 1024)
        
        final_predictions = model(test_input)
        
        # Should handle larger configurations
        expected_shape = (2, 12, 192)  # batch=2, variables=12, prediction_length=192
        
        assert final_predictions.shape == expected_shape


class TestPSformerConfig:
    """Test PSformerConfig validation"""
    
    def test_valid_configuration(self):
        """Test that valid configuration is accepted"""
        config = PSformerConfig(
            sequence_length=512,
            num_variables=7,
            patch_size=32,
            num_encoder_layers=3,
            prediction_length=96
        )
        
        assert config.sequence_length == 512
        assert config.num_variables == 7
        assert config.patch_size == 32
        assert config.num_encoder_layers == 3
        assert config.prediction_length == 96
    
    def test_invalid_sequence_length_not_divisible_by_patch_size(self):
        """Test that invalid sequence length raises error"""
        with pytest.raises(ValueError, match="must be divisible by patch size"):
            PSformerConfig(
                sequence_length=513,  # Not divisible by 32
                num_variables=7,
                patch_size=32,
                num_encoder_layers=3,
                prediction_length=96
            )
    
    def test_invalid_negative_values(self):
        """Test that negative values raise errors"""
        with pytest.raises(ValueError, match="must be positive"):
            PSformerConfig(
                sequence_length=512,
                num_variables=-1,  # Invalid
                patch_size=32,
                num_encoder_layers=3,
                prediction_length=96
            )
        
        with pytest.raises(ValueError, match="must be positive"):
            PSformerConfig(
                sequence_length=512,
                num_variables=7,
                patch_size=-1,  # Invalid
                num_encoder_layers=3,
                prediction_length=96
            )
        
        with pytest.raises(ValueError, match="must be positive"):
            PSformerConfig(
                sequence_length=512,
                num_variables=7,
                patch_size=32,
                num_encoder_layers=0,  # Invalid
                prediction_length=96
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
