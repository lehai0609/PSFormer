import pytest
import torch
import numpy as np
from src.blocks.RevIN import RevIN


class TestRevINInputValidation:
    """Data validation tests for RevIN"""

    def test_input_shape_validation(self):
        """Test input shape validation"""
        num_features = 32
        revin = RevIN(num_features)
        
        # Valid input shape [batch, channels, sequence_length]
        valid_input = torch.randn(16, num_features, 112)
        normalized = revin(valid_input, mode='norm')
        assert normalized.shape == valid_input.shape
        
        # Invalid 1D tensor
        with pytest.raises(IndexError):
            invalid_input = torch.randn(num_features)
            revin(invalid_input, mode='norm')
            
        # Invalid 2D tensor
        with pytest.raises(IndexError):
            invalid_input = torch.randn(16, num_features)
            revin(invalid_input, mode='norm')
            
        # Invalid 4D tensor
        with pytest.raises(IndexError):
            invalid_input = torch.randn(16, num_features, 112, 2)
            revin(invalid_input, mode='norm')

    def test_input_data_types(self):
        """Test input data type handling"""
        num_features = 32
        revin = RevIN(num_features)
        
        # Valid float32 input
        input_float32 = torch.randn(16, num_features, 112, dtype=torch.float32)
        normalized = revin(input_float32, mode='norm')
        assert normalized.dtype == torch.float32
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()
        
        # Valid float64 input
        input_float64 = torch.randn(16, num_features, 112, dtype=torch.float64)
        normalized = revin(input_float64, mode='norm')
        assert normalized.dtype == torch.float64
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()
        
        # Integer input should work (converted to float)
        input_int = torch.randint(0, 100, (16, num_features, 112))
        normalized = revin(input_int.float(), mode='norm')  # Convert to float first
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()

    def test_statistical_prerequisites(self):
        """Test handling of edge case data"""
        num_features = 32
        revin = RevIN(num_features)
        
        # All zeros
        zeros_input = torch.zeros(16, num_features, 112)
        normalized = revin(zeros_input, mode='norm')
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()
        
        # Constant values
        constant_input = torch.full((16, num_features, 112), 5.0)
        normalized = revin(constant_input, mode='norm')
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()
        
        # Extreme outliers
        extreme_input = torch.randn(16, num_features, 112)
        extreme_input[0, 0, 0] = 1e10
        extreme_input[1, 1, 1] = -1e10
        normalized = revin(extreme_input, mode='norm')
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()


class TestRevINProcessingLogic:
    """Processing/Feature engineering tests for RevIN"""

    def test_instance_statistics_computation(self):
        """Test instance statistics computation"""
        num_features = 32
        revin = RevIN(num_features)
        
        # Create input with known statistics
        input_tensor = torch.randn(16, num_features, 112)
        
        # Compute manually
        dim2reduce = tuple(range(1, input_tensor.ndim-1))  # Should be (1,) for 3D tensor
        manual_mean = torch.mean(input_tensor, dim=dim2reduce, keepdim=True)
        manual_stdev = torch.sqrt(torch.var(input_tensor, dim=dim2reduce, keepdim=True, unbiased=False) + revin.eps)
        
        # Run through RevIN to compute statistics
        revin(input_tensor, mode='norm')  # This should compute and store mean/stdev
        
        # Check if statistics match
        assert torch.allclose(revin.mean, manual_mean, atol=1e-6)
        assert torch.allclose(revin.stdev, manual_stdev, atol=1e-6)

    def test_normalization_transformation(self):
        """Test normalization transformation"""
        num_features = 32
        revin = RevIN(num_features)
        
        # Create input with known statistics
        input_tensor = torch.randn(16, num_features, 112)
        
        # Manual normalization
        dim2reduce = tuple(range(1, input_tensor.ndim-1))
        mean = torch.mean(input_tensor, dim=dim2reduce, keepdim=True)
        stdev = torch.sqrt(torch.var(input_tensor, dim=dim2reduce, keepdim=True, unbiased=False) + revin.eps)
        manual_normalized = (input_tensor - mean) / stdev
        
        # Run through RevIN
        revin_normalized = revin(input_tensor, mode='norm')
        
        # Without affine transformation, they should be equal
        revin_no_affine = RevIN(num_features, affine=False)
        revin_no_affine_normalized = revin_no_affine(input_tensor, mode='norm')
        
        assert torch.allclose(revin_no_affine_normalized, manual_normalized, atol=1e-6)

    def test_affine_parameter_application(self):
        """Test affine parameter application"""
        num_features = 32
        revin = RevIN(num_features)
        
        # Create input
        input_tensor = torch.randn(16, num_features, 112)
        
        # Manual affine transformation
        dim2reduce = tuple(range(1, input_tensor.ndim-1))
        mean = torch.mean(input_tensor, dim=dim2reduce, keepdim=True)
        stdev = torch.sqrt(torch.var(input_tensor, dim=dim2reduce, keepdim=True, unbiased=False) + revin.eps)
        normalized = (input_tensor - mean) / stdev
        manual_affine = normalized * revin.affine_weight + revin.affine_bias
        
        # Run through RevIN
        revin_result = revin(input_tensor, mode='norm')
        
        # They should be equal
        assert torch.allclose(revin_result, manual_affine, atol=1e-6)

    def test_denormalization_symmetry(self):
        """Test denormalization symmetry"""
        num_features = 32
        revin = RevIN(num_features)
        
        # Create input
        input_tensor = torch.randn(16, num_features, 112)
        
        # Normalize
        normalized = revin(input_tensor, mode='norm')
        
        # Denormalize
        denormalized = revin(normalized, mode='denorm')
        
        # Should be equal to original (within numerical precision)
        assert torch.allclose(denormalized, input_tensor, atol=1e-5)


class TestRevINModelIntegration:
    """Model behavior tests for RevIN"""

    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline"""
        num_features = 32
        seq_length = 112
        revin = RevIN(num_features)
        
        # Sample time series data
        input_tensor = torch.randn(16, num_features, seq_length)
        
        # Full pipeline: norm -> denorm
        normalized = revin(input_tensor, mode='norm')
        output = revin(normalized, mode='denorm')
        
        # Output shape should match input
        assert output.shape == input_tensor.shape
        
        # And should be close to original
        assert torch.allclose(output, input_tensor, atol=1e-5)

    def test_gradient_flow(self):
        """Test gradient flow through RevIN"""
        num_features = 32
        revin = RevIN(num_features)
        
        # Input with gradients
        input_tensor = torch.randn(16, num_features, 112, requires_grad=True)
        
        # Forward pass
        normalized = revin(input_tensor, mode='norm')
        loss = normalized.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist for affine parameters
        if revin.affine:
            assert revin.affine_weight.grad is not None
            assert revin.affine_bias.grad is not None
            assert not torch.all(revin.affine_weight.grad == 0)
            assert not torch.all(revin.affine_bias.grad == 0)

    def test_parameter_learning(self):
        """Test parameter learning"""
        num_features = 32
        revin = RevIN(num_features)
        
        # Store initial parameters
        initial_weight = revin.affine_weight.clone().detach()
        initial_bias = revin.affine_bias.clone().detach()
        
        # Simple optimization loop
        optimizer = torch.optim.SGD([revin.affine_weight, revin.affine_bias], lr=0.01)
        
        for _ in range(5):
            input_tensor = torch.randn(16, num_features, 112, requires_grad=True)
            normalized = revin(input_tensor, mode='norm')
            loss = normalized.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Parameters should have changed
        assert not torch.allclose(revin.affine_weight, initial_weight)
        assert not torch.allclose(revin.affine_bias, initial_bias)

    def test_mode_consistency(self):
        """Test mode consistency"""
        num_features = 32
        revin = RevIN(num_features)
        
        # Same input
        input_tensor = torch.randn(16, num_features, 112)
        
        # Process in both modes (norm then denorm)
        normalized = revin(input_tensor, mode='norm')
        denormalized = revin(normalized, mode='denorm')
        
        # Should be consistent regardless of mode
        assert torch.allclose(denormalized, input_tensor, atol=1e-5)


class TestRevINRobustness:
    """Robustness tests for RevIN"""

    def test_zero_variance_handling(self):
        """Test zero variance handling"""
        num_features = 32
        revin = RevIN(num_features)
        
        # Input sequences with constant values (σ = 0)
        constant_input = torch.full((16, num_features, 112), 5.0)
        
        # Should not crash
        normalized = revin(constant_input, mode='norm')
        denormalized = revin(normalized, mode='denorm')
        
        # Should be reasonable values (not NaN/Inf)
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()
        assert not torch.isnan(denormalized).any()
        assert not torch.isinf(denormalized).any()

    def test_extreme_value_handling(self):
        """Test extreme value handling"""
        num_features = 32
        revin = RevIN(num_features)
        
        # Inputs with very large/small values
        extreme_input = torch.randn(16, num_features, 112) * 1e6
        
        # Should not crash and maintain numerical stability
        normalized = revin(extreme_input, mode='norm')
        denormalized = revin(normalized, mode='denorm')
        
        # Should be finite values
        assert torch.isfinite(normalized).all()
        assert torch.isfinite(denormalized).all()

    def test_batch_independence(self):
        """Test batch independence"""
        num_features = 32
        revin = RevIN(num_features)
        
        # Batched input with different statistical properties
        batch_input = torch.randn(16, num_features, 112)
        
        # Process all at once
        batch_normalized = revin(batch_input, mode='norm')
        
        # Process individually
        individual_results = []
        for i in range(batch_input.shape[0]):
            single_input = batch_input[i:i+1, :, :]  # Keep batch dimension
            single_normalized = revin(single_input, mode='norm')
            individual_results.append(single_normalized)
        
        # Concatenate individual results
        concatenated = torch.cat(individual_results, dim=0)
        
        # Should be equal (within numerical precision)
        assert torch.allclose(batch_normalized, concatenated, atol=1e-5)

    def test_different_sequence_lengths(self):
        """Test different sequence lengths"""
        num_features = 32
        revin = RevIN(num_features)
        
        # Different sequence lengths
        seq_lengths = [56, 112, 224]
        
        for seq_len in seq_lengths:
            input_tensor = torch.randn(16, num_features, seq_len)
            normalized = revin(input_tensor, mode='norm')
            denormalized = revin(normalized, mode='denorm')
            
            # Check shapes
            assert normalized.shape == input_tensor.shape
            assert denormalized.shape == input_tensor.shape
            
            # Check correctness
            assert torch.allclose(denormalized, input_tensor, atol=1e-5)


class TestRevINArchitecture:
    """Architecture tests for RevIN integration with PSformer"""

    def test_parameter_count_verification(self):
        """Test parameter count verification"""
        num_features = 32
        revin_affine = RevIN(num_features, affine=True)
        revin_no_affine = RevIN(num_features, affine=False)
        
        # With affine: 2 parameters of shape (num_features,)
        affine_params = sum(p.numel() for p in revin_affine.parameters())
        expected_affine = 2 * num_features
        assert affine_params == expected_affine
        
        # Without affine: 0 parameters
        no_affine_params = sum(p.numel() for p in revin_no_affine.parameters())
        assert no_affine_params == 0

    def test_weight_shapes(self):
        """Test weight shapes"""
        num_features = 32
        revin = RevIN(num_features, affine=True)
        
        # Check parameter shapes
        assert revin.affine_weight.shape == (num_features,)
        assert revin.affine_bias.shape == (num_features,)

    def test_device_placement_consistency(self):
        """Test device placement consistency"""
        num_features = 32
        revin = RevIN(num_features)
        
        # Test on CPU
        input_cpu = torch.randn(16, num_features, 112)
        revin_cpu = RevIN(num_features)
        output_cpu = revin_cpu(input_cpu, mode='norm')
        assert output_cpu.device == input_cpu.device
        
        # Test on GPU if available
        if torch.cuda.is_available():
            input_cuda = input_cpu.cuda()
            revin_cuda = RevIN(num_features).cuda()
            output_cuda = revin_cuda(input_cuda, mode='norm')
            assert output_cuda.device == input_cuda.device


# Additional comparison and validation tests could be added here if needed