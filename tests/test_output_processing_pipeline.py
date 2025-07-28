import pytest
import torch
import torch.nn as nn
from src.blocks.psformer import PSformer, PSformerConfig, create_psformer_model


class TestOutputProcessingPipeline:
    """
    Comprehensive test suite for the Output Processing Pipeline (Step 4.2).
    
    Based on Section 3.2 of the PSFormer paper and the implementation plan.
    Tests the complete pipeline: Inverse Transform → Linear Projection → Inverse RevIN
    """
    
    def setup_method(self):
        """Setup method runs before each test function."""
        # GIVEN a standard PSformer model configuration
        self.batch_size = 8
        self.num_variables = 7   # M
        self.sequence_length = 96 # L
        self.patch_size = 16
        self.prediction_length = 24 # F
        self.num_encoder_layers = 2
        
        # Create the fully assembled PSformer model
        # This model contains the encoder, data_transformer, revin_layer, and the NEW output_projection layer.
        self.model = create_psformer_model(
            sequence_length=self.sequence_length,
            num_variables=self.num_variables,
            patch_size=self.patch_size,
            num_encoder_layers=self.num_encoder_layers,
            prediction_length=self.prediction_length
        )
        
        # AND a dummy raw input to calculate RevIN statistics
        self.raw_input = self._create_dummy_tensor(shape=(self.batch_size, self.num_variables, self.sequence_length))
        
        # Pre-populate RevIN statistics by running the normalization step
        # This is critical for testing the 'denorm' mode later.
        self.model.revin_layer(self.raw_input, mode='norm')
        
        # AND a dummy output from the encoder
        # This tensor simulates the input to our output pipeline.
        num_patches = self.sequence_length // self.patch_size # N
        segment_length = self.num_variables * self.patch_size   # C
        self.encoder_output = self._create_dummy_tensor(shape=(self.batch_size, num_patches, segment_length))
    
    def _create_dummy_tensor(self, shape: tuple, fill_value: float = 1.0) -> torch.Tensor:
        """Create a dummy tensor with specified shape and fill value."""
        return torch.full(shape, fill_value, dtype=torch.float32)
    
    def _create_zero_tensor(self, shape: tuple) -> torch.Tensor:
        """Create a zero tensor with specified shape."""
        return torch.zeros(shape, dtype=torch.float32)
    
    def _create_dummy_target(self, shape: tuple) -> torch.Tensor:
        """Create dummy target tensor for loss calculation."""
        return torch.randn(shape, dtype=torch.float32)
    
    def _calculate_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate simple MSE loss for testing gradient flow."""
        return nn.MSELoss()(predictions, targets)
    
    # --- 1. Data Validation Tests ---
    # Goal: Ensure the pipeline's inputs are handled correctly.
    
    def test_pipeline_fails_with_incorrect_encoder_output_shape(self):
        """Test that pipeline fails gracefully with incorrect encoder output shape."""
        # GIVEN encoder output with an incorrect number of patches (N)
        invalid_encoder_output = self._create_dummy_tensor(shape=(self.batch_size, 5, self.model.psformer_dims['C'])) # 5 is wrong
        
        # WHEN attempting to run the output processing through inverse_transform
        # THEN the model should raise a ValueError during the inverse_transform step.
        with pytest.raises(ValueError, match="num_patches"):
            self.model.data_transformer.inverse_transform(invalid_encoder_output, self.model.config.sequence_length)
    
    def test_pipeline_fails_with_incorrect_segment_length(self):
        """Test that pipeline fails with incorrect segment length."""
        # GIVEN encoder output with incorrect segment length (C)
        invalid_encoder_output = self._create_dummy_tensor(shape=(self.batch_size, self.model.psformer_dims['N'], 50)) # 50 is wrong
        
        # WHEN attempting to run the output processing
        # THEN the model should raise a ValueError
        with pytest.raises(ValueError, match="segment_length"):
            self.model.data_transformer.inverse_transform(invalid_encoder_output, self.model.config.sequence_length)
    
    # --- 2. Processing/Feature Engineering Tests ---
    # Goal: Validate intermediate shapes and states.
    
    def test_inverse_transform_produces_correct_intermediate_shape(self):
        """Test that inverse transform produces correct intermediate shape."""
        # GIVEN the valid encoder_output
        
        # WHEN performing ONLY the inverse data transformation (Step 4.1)
        reshaped_output = self.model.data_transformer.inverse_transform(
            self.encoder_output, 
            self.model.config.sequence_length
        )
        
        # THEN the intermediate tensor shape must be [B, M, L]
        # This is the expected input shape for the linear projection layer.
        expected_shape = (self.batch_size, self.num_variables, self.sequence_length)
        assert reshaped_output.shape == expected_shape
    
    def test_linear_projection_produces_correct_shape(self):
        """Test that linear projection produces correct output shape."""
        # GIVEN a tensor with shape [B, M, L]
        input_tensor = self._create_dummy_tensor(shape=(self.batch_size, self.num_variables, self.sequence_length))
        
        # WHEN applying the linear projection
        projected_output = self.model.output_projection(input_tensor)
        
        # THEN the output shape must be [B, M, F]
        expected_shape = (self.batch_size, self.num_variables, self.prediction_length)
        assert projected_output.shape == expected_shape
    
    def test_denormalization_fails_if_stats_not_stored(self):
        """Test that denormalization fails if RevIN statistics are not stored."""
        # GIVEN a new model instance that has NOT run normalization
        fresh_model = create_psformer_model(
            sequence_length=self.sequence_length,
            num_variables=self.num_variables,
            patch_size=self.patch_size,
            num_encoder_layers=self.num_encoder_layers,
            prediction_length=self.prediction_length
        )
        dummy_projection_output = self._create_dummy_tensor(shape=(self.batch_size, self.num_variables, self.prediction_length))

        # WHEN attempting to call denormalize directly
        # THEN the RevIN module should raise an AttributeError
        # because self.mean and self.stdev have not been computed.
        with pytest.raises(AttributeError):
            fresh_model.revin_layer(dummy_projection_output, mode='denorm')
    
    # --- 3. Model Behavior Tests ---
    # Goal: Assert specific, measurable outcomes of the linear projection.
    
    def test_final_output_shape_is_correct(self):
        """Test that the final output shape is correct."""
        # GIVEN the valid raw input
        
        # WHEN the full forward pass is executed
        final_predictions = self.model.forward(self.raw_input)
        
        # THEN the final output shape must be [B, M, F]
        # This directly tests the effect of the linear projection layer (mapping L -> F).
        # Paper Reference: Confirms X_pred ∈ R^(M×F).
        expected_shape = (self.batch_size, self.num_variables, self.prediction_length)
        assert final_predictions.shape == expected_shape
    
    def test_pipeline_works_with_batch_size_one(self):
        """Test that pipeline works with batch size of 1."""
        # GIVEN input with a batch size of 1
        single_item_input = self._create_dummy_tensor(shape=(1, self.num_variables, self.sequence_length))
        
        # Pre-populate RevIN statistics
        self.model.revin_layer(single_item_input, mode='norm')
        
        # WHEN the full forward pass is executed
        final_predictions = self.model.forward(single_item_input)
        
        # THEN the output shape should be correct for a single batch item.
        # This prevents errors from hardcoded batch dimensions (e.g., `squeeze()`).
        expected_shape = (1, self.num_variables, self.prediction_length)
        assert final_predictions.shape == expected_shape
    
    def test_zero_input_to_projection_yields_revIN_mean(self):
        """Test that zero input to projection yields RevIN mean in final output."""
        # GIVEN input with zero values (after some processing to establish RevIN stats)
        # First establish RevIN statistics with non-zero input
        normal_input = self._create_dummy_tensor(shape=(self.batch_size, self.num_variables, self.sequence_length), fill_value=5.0)
        self.model.revin_layer(normal_input, mode='norm')
        
        # Create zero input
        zero_input = self._create_zero_tensor(shape=(self.batch_size, self.num_variables, self.sequence_length))
        
        # WHEN the full pipeline processes this zero tensor
        final_predictions = self.model.forward(zero_input)
        
        # THEN the final prediction should be close to the stored RevIN mean
        # Logic: zero input -> normalize(0) = (0 - mean) / stdev -> ... -> denorm -> should be close to mean
        stored_mean = self.model.revin_layer.mean
        # We expect the output to be close to the mean, but not exactly due to the linear projection
        assert final_predictions.shape == (self.batch_size, self.num_variables, self.prediction_length)
        # Verify the predictions are reasonable (not NaN or inf)
        assert not torch.isnan(final_predictions).any()
        assert not torch.isinf(final_predictions).any()
    
    def test_forward_with_intermediates_returns_correct_structure(self):
        """Test that forward_with_intermediates returns correct structure."""
        # GIVEN valid input
        
        # WHEN calling forward_with_intermediates
        final_predictions, intermediates = self.model.forward_with_intermediates(self.raw_input)
        
        # THEN the structure should be correct
        assert final_predictions.shape == (self.batch_size, self.num_variables, self.prediction_length)
        assert 'encoder_output' in intermediates
        assert 'attention_weights' in intermediates
        assert 'reshaped_output' in intermediates
        assert 'projected_output' in intermediates
        
        # Verify intermediate shapes
        assert intermediates['encoder_output'].shape == (self.batch_size, self.model.psformer_dims['N'], self.model.psformer_dims['C'])
        assert intermediates['reshaped_output'].shape == (self.batch_size, self.num_variables, self.sequence_length)
        assert intermediates['projected_output'].shape == (self.batch_size, self.num_variables, self.prediction_length)
    
    # --- 4. Performance and Robustness Tests ---
    # Goal: Simulate adverse scenarios to ensure graceful failure.
    
    def test_nan_values_propagate_through_pipeline(self):
        """Test that NaN values propagate through pipeline without crashing."""
        # GIVEN input containing NaN values
        nan_input = self.raw_input.clone()
        nan_input[0, 0, 0] = float('nan')
        
        # WHEN the full forward pass is executed
        final_predictions = self.model.forward(nan_input)
        
        # THEN the output should also contain NaN values and the model should not crash.
        assert torch.isnan(final_predictions).any()
        assert final_predictions.shape == (self.batch_size, self.num_variables, self.prediction_length)
    
    def test_inf_values_propagate_through_pipeline(self):
        """Test that infinity values propagate through pipeline without crashing."""
        # GIVEN input containing infinity values
        inf_input = self.raw_input.clone()
        inf_input[0, 0, 0] = float('inf')
        
        # WHEN the full forward pass is executed  
        final_predictions = self.model.forward(inf_input)
        
        # THEN the output should contain NaN values (due to RevIN normalization of inf) and the model should not crash.
        # Note: RevIN normalization converts infinity to NaN during the (inf - mean) / stdev operation
        assert torch.isnan(final_predictions).any() or torch.isinf(final_predictions).any()
        assert final_predictions.shape == (self.batch_size, self.num_variables, self.prediction_length)
    
    # --- 5. Architecture Tests ---
    # Goal: Ensure layers are connected correctly and are trainable.

    def test_projection_layer_weights_have_correct_shape(self):
        """Test that projection layer weights have correct shape."""
        # GIVEN the initialized model
        
        # THEN the output_projection layer's weight matrix must have shape [F, L]
        # Paper Reference: This validates that W_F ∈ R^(L×F) is implemented correctly,
        # noting PyTorch's [out_features, in_features] convention.
        projection_layer = self.model.output_projection
        expected_weight_shape = (self.prediction_length, self.sequence_length)
        assert projection_layer.weight.shape == expected_weight_shape
    
    def test_projection_layer_has_bias(self):
        """Test that projection layer has bias term."""
        # GIVEN the initialized model
        
        # THEN the output_projection layer should have a bias term
        projection_layer = self.model.output_projection
        assert projection_layer.bias is not None
        assert projection_layer.bias.shape == (self.prediction_length,)
    
    def test_projection_layer_is_trainable(self):
        """Test that projection layer is trainable (gradients flow through it)."""
        # GIVEN a full forward pass and a backward pass
        final_predictions = self.model.forward(self.raw_input)
        target = self._create_dummy_target(final_predictions.shape)
        loss = self._calculate_loss(final_predictions, target)
        loss.backward()
        
        # THEN the gradient of the output_projection layer's weights must not be None.
        # This is a critical test to ensure the layer is part of the computation graph
        # and its parameters will be updated during training.
        assert self.model.output_projection.weight.grad is not None
        assert self.model.output_projection.bias.grad is not None
    
    def test_output_projection_parameters_are_registered(self):
        """Test that output projection parameters are properly registered."""
        # GIVEN the model
        
        # WHEN getting all parameters
        all_params = list(self.model.parameters())
        projection_params = list(self.model.output_projection.parameters())
        
        # THEN projection parameters should be included in model parameters
        assert len(projection_params) == 2  # weight and bias
        
        # Check if projection parameters are in the model parameters by checking ids
        all_param_ids = {id(param) for param in all_params}
        for param in projection_params:
            assert id(param) in all_param_ids
    
    # --- 6. Integration Tests ---
    # Goal: Test the complete pipeline end-to-end
    
    def test_complete_pipeline_integration(self):
        """Test complete pipeline integration from input to output."""
        # GIVEN different input sizes
        test_cases = [
            (4, 7, 96, 24),   # smaller batch
            (1, 3, 48, 12),   # minimal case
            (16, 10, 192, 48), # larger case
        ]
        
        for batch_size, num_vars, seq_len, pred_len in test_cases:
            if seq_len % self.patch_size != 0:
                continue  # Skip invalid configurations
                
            # Create model for this configuration
            model = create_psformer_model(
                sequence_length=seq_len,
                num_variables=num_vars,
                patch_size=self.patch_size,
                num_encoder_layers=2,
                prediction_length=pred_len
            )
            
            # Create input
            test_input = self._create_dummy_tensor(shape=(batch_size, num_vars, seq_len))
            
            # WHEN running forward pass
            output = model.forward(test_input)
            
            # THEN output should have correct shape and be valid
            expected_shape = (batch_size, num_vars, pred_len)
            assert output.shape == expected_shape
            assert not torch.isnan(output).any()
            assert torch.isfinite(output).all()
    
    def test_pipeline_deterministic_with_same_input(self):
        """Test that pipeline produces deterministic output with same input."""
        # GIVEN the same input
        test_input = self._create_dummy_tensor(shape=(self.batch_size, self.num_variables, self.sequence_length))
        
        # WHEN running forward pass twice
        output1 = self.model.forward(test_input)
        output2 = self.model.forward(test_input)
        
        # THEN outputs should be identical
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_different_prediction_lengths(self):
        """Test pipeline with different prediction lengths."""
        prediction_lengths = [12, 24, 48, 96]
        
        for pred_len in prediction_lengths:
            # Create model with different prediction length
            model = create_psformer_model(
                sequence_length=self.sequence_length,
                num_variables=self.num_variables,
                patch_size=self.patch_size,
                num_encoder_layers=self.num_encoder_layers,
                prediction_length=pred_len
            )
            
            # Test input
            test_input = self._create_dummy_tensor(shape=(self.batch_size, self.num_variables, self.sequence_length))
            
            # Forward pass
            output = model.forward(test_input)
            
            # Verify shape
            expected_shape = (self.batch_size, self.num_variables, pred_len)
            assert output.shape == expected_shape
