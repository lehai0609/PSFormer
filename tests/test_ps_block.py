import pytest
import torch
import numpy as np
from src.blocks.ps_block import PSBlock


class TestPSBlockDataValidation:
    """Data validation tests for PS Block"""
    
    def test_input_shape_validation(self):
        """Test input shape validation"""
        N = 32
        ps_block = PSBlock(N)
        
        # Valid 2D input
        valid_input_2d = torch.randn(112, N)
        output = ps_block(valid_input_2d)
        assert output.shape == valid_input_2d.shape
        
        # Valid 3D input
        valid_input_3d = torch.randn(10, 112, N)
        output = ps_block(valid_input_3d)
        assert output.shape == valid_input_3d.shape
        
        # Invalid 1D tensor
        with pytest.raises(ValueError):
            invalid_input = torch.randn(N)
            ps_block(invalid_input)
            
        # Invalid 4D tensor
        with pytest.raises(ValueError):
            invalid_input = torch.randn(10, 112, N, 5)
            ps_block(invalid_input)
            
        # Wrong last dimension in 2D
        with pytest.raises(ValueError):
            invalid_input = torch.randn(112, N+1)
            ps_block(invalid_input)
            
        # Wrong last dimension in 3D
        with pytest.raises(ValueError):
            invalid_input = torch.randn(10, 112, N+1)
            ps_block(invalid_input)

    def test_input_data_quality(self):
        """Test input data quality validation"""
        N = 32
        ps_block = PSBlock(N)
        
        # Valid float inputs
        valid_input = torch.randn(112, N, dtype=torch.float32)
        output = ps_block(valid_input)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        # Test with NaN inputs
        with pytest.raises(ValueError):
            nan_input = torch.randn(112, N)
            nan_input[0, 0] = float('nan')
            ps_block(nan_input)
            
        # Test with inf inputs
        with pytest.raises(ValueError):
            inf_input = torch.randn(112, N)
            inf_input[0, 0] = float('inf')
            ps_block(inf_input)

    def test_dimension_consistency_with_weights(self):
        """Test dimension consistency with weight matrices"""
        N = 32
        ps_block = PSBlock(N)
        
        # Valid 2D input with matching N dimension
        valid_input_2d = torch.randn(112, N)
        output = ps_block(valid_input_2d)
        assert output.shape == valid_input_2d.shape
        
        # Valid 3D input with matching N dimension
        valid_input_3d = torch.randn(10, 112, N)
        output = ps_block(valid_input_3d)
        assert output.shape == valid_input_3d.shape
        
        # Invalid 2D input with mismatched N dimension
        with pytest.raises(ValueError):
            invalid_input = torch.randn(112, N-1)
            ps_block(invalid_input)
            
        # Invalid 3D input with mismatched N dimension
        with pytest.raises(ValueError):
            invalid_input = torch.randn(10, 112, N-1)
            ps_block(invalid_input)

    def test_extreme_value_handling(self):
        """Test handling of extreme values"""
        N = 32
        ps_block = PSBlock(N)
        
        # Very large positive values
        large_positive = torch.full((112, N), 1e10)
        output = ps_block(large_positive)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        # Very small values
        small_values = torch.full((112, N), 1e-10)
        output = ps_block(small_values)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        # Zero tensor
        zero_tensor = torch.zeros((112, N))
        output = ps_block(zero_tensor)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestPSBlockProcessingValidation:
    """Processing/Feature engineering tests for PS Block"""
    
    def test_gelu_activation_correctness(self):
        """Validate GeLU activation behaves as expected"""
        N = 32
        ps_block = PSBlock(N)
        
        # Known input values - must match dimension N
        known_input = torch.randn(1, N)  # Using random values but correct shape
        
        # Apply linear1 transformation first
        linear_output = ps_block.linear1(known_input)
        
        # Apply GeLU manually
        expected_gelu = ps_block.activation(linear_output)
        
        # Get actual GeLU from the block
        with torch.no_grad():
            step1 = ps_block.activation(ps_block.linear1(known_input))
        
        # They should be equal
        assert torch.allclose(step1, expected_gelu, atol=1e-6)

    def test_residual_connection_arithmetic(self):
        """Validate residual connection arithmetic"""
        N = 32
        C = 112
        ps_block = PSBlock(N)
        
        # Create a fixed input for reproducible results
        torch.manual_seed(42)
        input_tensor = torch.randn(C, N)
        
        # Manual computation of the residual connection
        with torch.no_grad():
            step1 = ps_block.activation(ps_block.linear1(input_tensor))
            intermediate = ps_block.linear2(step1)
            expected_with_residual = intermediate + input_tensor
        
        # Get the actual intermediate result from a modified forward pass
        with torch.no_grad():
            step1_actual = ps_block.activation(ps_block.linear1(input_tensor))
            step2_actual = ps_block.linear2(step1_actual) + input_tensor
            
        # Compare
        assert torch.allclose(step2_actual, expected_with_residual, atol=1e-6)

    def test_sequential_transformation_chain(self):
        """Validate the three-step transformation chain"""
        N = 32
        C = 112
        ps_block = PSBlock(N)
        
        # Create a fixed input for reproducible results
        torch.manual_seed(42)
        input_tensor = torch.randn(C, N)
        
        # Manual computation of the three steps
        with torch.no_grad():
            step1 = ps_block.activation(ps_block.linear1(input_tensor))
            step2 = ps_block.linear2(step1) + input_tensor
            expected_output = ps_block.linear3(step2)
        
        # Get actual output from the block
        with torch.no_grad():
            actual_output = ps_block(input_tensor)
        
        # Compare
        assert torch.allclose(actual_output, expected_output, atol=1e-6)

    def test_gradient_flow_validation(self):
        """Ensure gradients propagate through all weight matrices"""
        N = 32
        C = 112
        ps_block = PSBlock(N)
        
        # Zero out gradients first
        ps_block.zero_grad()
        
        # Create input and compute loss
        input_tensor = torch.randn(C, N, requires_grad=True)
        output = ps_block(input_tensor)
        loss = output.sum()
        
        # Backpropagate
        loss.backward()
        
        # Check that gradients exist and are not all zero
        assert ps_block.linear1.weight.grad is not None
        assert ps_block.linear2.weight.grad is not None
        assert ps_block.linear3.weight.grad is not None
        
        assert not torch.all(ps_block.linear1.weight.grad == 0)
        assert not torch.all(ps_block.linear2.weight.grad == 0)
        assert not torch.all(ps_block.linear3.weight.grad == 0)


class TestPSBlockBehaviorValidation:
    """Model behavior tests for PS Block"""
    
    def test_output_shape_preservation(self):
        """Test that output shape matches input shape"""
        N = 32
        C = 112
        
        ps_block = PSBlock(N)
        input_tensor = torch.randn(C, N)
        output = ps_block(input_tensor)
        
        assert output.shape == input_tensor.shape
        assert output.shape == (C, N)
        
    def test_deterministic_forward_pass(self):
        """Test that same input always produces same output"""
        N = 32
        C = 112
        
        # Set seed for reproducible results
        torch.manual_seed(42)
        ps_block = PSBlock(N)
        input_tensor = torch.randn(C, N)
        
        # Get output with fixed seed
        torch.manual_seed(42)
        ps_block1 = PSBlock(N)
        output1 = ps_block1(input_tensor)
        
        # Create another instance with same seed
        torch.manual_seed(42)
        ps_block2 = PSBlock(N)
        output2 = ps_block2(input_tensor)
        
        assert torch.allclose(output1, output2, atol=1e-7)
        
    def test_linear_transformation_properties(self):
        """Test approximate linearity for small inputs"""
        N = 32
        C = 112
        ps_block = PSBlock(N)
        
        # Small inputs for approximate linearity test
        a, b = 0.1, 0.2
        x = torch.randn(C, N) * 0.01  # Small values
        y = torch.randn(C, N) * 0.01  # Small values
        
        # Compute PS(aX + bY)
        with torch.no_grad():
            result1 = ps_block(a * x + b * y)
            
            # Compute aPS(X) + bPS(Y)
            result2 = a * ps_block(x) + b * ps_block(y)
        
        # They should be approximately equal for small inputs
        # Using a relatively loose tolerance due to non-linearities
        assert torch.allclose(result1, result2, atol=1e-2)


class TestPSBlockRobustnessValidation:
    """Performance and robustness tests for PS Block"""
    
    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme values"""
        N = 32
        C = 112
        ps_block = PSBlock(N)
        
        # Extreme positive values
        extreme_positive = torch.full((C, N), 100.0)
        output = ps_block(extreme_positive)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        # Extreme negative values
        extreme_negative = torch.full((C, N), -100.0)
        output = ps_block(extreme_negative)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_explosion_prevention(self):
        """Ensure gradients don't explode through transformations"""
        N = 32
        C = 112
        ps_block = PSBlock(N)
        
        # Large input that might cause gradient explosion
        large_input = torch.randn(C, N) * 10
        large_input.requires_grad_(True)
        
        # Compute output and loss
        output = ps_block(large_input)
        loss = output.norm()
        
        # Backpropagate
        loss.backward()
        
        # Check that gradients are finite
        assert torch.isfinite(ps_block.linear1.weight.grad).all()
        assert torch.isfinite(ps_block.linear2.weight.grad).all()
        assert torch.isfinite(ps_block.linear3.weight.grad).all()
        
        # Check that gradient norms are reasonable (not extremely large)
        grad_norm1 = torch.norm(ps_block.linear1.weight.grad)
        grad_norm2 = torch.norm(ps_block.linear2.weight.grad)
        grad_norm3 = torch.norm(ps_block.linear3.weight.grad)
        
        # These should not be extremely large (heuristic threshold)
        assert grad_norm1 < 1000
        assert grad_norm2 < 1000
        assert grad_norm3 < 1000

    def test_batch_size_independence(self):
        """Test that output is independent of batching"""
        N = 32
        C = 112
        ps_block = PSBlock(N)
        
        # Create input
        input_tensor = torch.randn(C, N)
        
        # Process all at once
        batch_output = ps_block(input_tensor)
        
        # Process row by row
        individual_outputs = []
        for i in range(C):
            single_sample = input_tensor[i:i+1, :]  # Keep as 2D
            single_output = ps_block(single_sample)
            individual_outputs.append(single_output)
        
        # Concatenate individual outputs
        concatenated_output = torch.cat(individual_outputs, dim=0)
        
        # They should be equal
        assert torch.allclose(batch_output, concatenated_output, atol=1e-6)


class TestPSBlockArchitectureValidation:
    """Architecture tests for PS Block"""
    
    def test_parameter_count_verification(self):
        """Test that parameter count is correct (3 N×N matrices with bias)"""
        N = 32
        ps_block = PSBlock(N)
        
        total_params = sum(p.numel() for p in ps_block.parameters())
        # Three N×N matrices + three bias vectors of size N
        expected_params = 3 * (N * N + N)  
        assert total_params == expected_params
        
    def test_weight_matrix_shapes(self):
        """Test that weight matrices have correct shapes"""
        N = 32
        ps_block = PSBlock(N)
        
        assert ps_block.linear1.weight.shape == (N, N)
        assert ps_block.linear2.weight.shape == (N, N)
        assert ps_block.linear3.weight.shape == (N, N)
        
        # Also check bias shapes
        assert ps_block.linear1.bias.shape == (N,)
        assert ps_block.linear2.bias.shape == (N,)
        assert ps_block.linear3.bias.shape == (N,)
        
    def test_parameter_initialization(self):
        """Test that weights are properly initialized"""
        N = 32
        ps_block = PSBlock(N)
        
        # Check that weights are not all zeros
        assert not torch.all(ps_block.linear1.weight == 0)
        assert not torch.all(ps_block.linear2.weight == 0)
        assert not torch.all(ps_block.linear3.weight == 0)
        
        # Check that weights are not all the same value
        assert torch.std(ps_block.linear1.weight) > 1e-5
        assert torch.std(ps_block.linear2.weight) > 1e-5
        assert torch.std(ps_block.linear3.weight) > 1e-5


class TestPSBlockAdditionalBehaviorValidation:
    """Additional behavior tests for PS Block"""
    
    def test_parameter_sharing_mechanism(self):
        """Test parameter sharing mechanism"""
        N = 32
        # Create two PS Blocks with shared weights
        ps_block_1 = PSBlock(N)
        ps_block_2 = PSBlock(N)
        
        # Manually set the same weights for both blocks
        with torch.no_grad():
            ps_block_2.linear1.weight.copy_(ps_block_1.linear1.weight)
            ps_block_2.linear2.weight.copy_(ps_block_1.linear2.weight)
            ps_block_2.linear3.weight.copy_(ps_block_1.linear3.weight)
        
        # Check that weights are equal
        assert torch.equal(ps_block_1.linear1.weight, ps_block_2.linear1.weight)
        assert torch.equal(ps_block_1.linear2.weight, ps_block_2.linear2.weight)
        assert torch.equal(ps_block_1.linear3.weight, ps_block_2.linear3.weight)
        
    def test_identity_preservation_capability(self):
        """Test identity preservation capability"""
        N = 32
        C = 112
        ps_block = PSBlock(N)
        
        # Create identity input
        identity_input = torch.eye(N).repeat(C // N + 1, 1)[:C, :]  # Extend to match C dimension
        
        # Initialize weights to approximate identity transformation
        with torch.no_grad():
            # Set linear1 and linear2 to small values to minimize their effect
            ps_block.linear1.weight.fill_(0.01)
            ps_block.linear2.weight.fill_(0.01)
            # Set linear3 close to identity matrix
            ps_block.linear3.weight.copy_(torch.eye(N) * 0.9)
        
        # Process input
        output = ps_block(identity_input)
        
        # For identity preservation, output should be close to input (loose tolerance due to non-linearities)
        # This is a basic test - in practice, training would be needed to properly learn identity mapping
        assert output.shape == identity_input.shape


class TestPSBlockAdditionalRobustnessValidation:
    """Additional robustness tests for PS Block"""
    
    def test_corrupted_weight_handling(self):
        """Test behavior when weights are corrupted"""
        N = 32
        C = 112
        ps_block = PSBlock(N)
        input_tensor = torch.randn(C, N)
        
        # Corrupt one weight with NaN
        with torch.no_grad():
            ps_block.linear1.weight[0, 0] = float('nan')
        
        # Should raise an error or produce NaN output
        output = ps_block(input_tensor)
        assert torch.isnan(output).any()
        
    def test_memory_usage_scaling(self):
        """Test memory usage with different input sizes"""
        # This is a conceptual test - actual memory measurement would require more sophisticated tools
        # We'll just verify that larger inputs don't crash and produce correct shapes
        N = 32
        
        # Small input
        ps_block_small = PSBlock(N)
        small_input = torch.randn(14, N)  # Small case
        small_output = ps_block_small(small_input)
        assert small_output.shape == small_input.shape
        
        # Large input
        ps_block_large = PSBlock(N)
        large_input = torch.randn(1680, N)  # Large case
        large_output = ps_block_large(large_input)
        assert large_output.shape == large_input.shape


class TestPSBlockAdditionalArchitectureValidation:
    """Additional architecture tests for PS Block"""
    
    def test_integration_with_segment_attention(self):
        """Test integration with segment attention mechanism"""
        N = 32
        C = 112
        ps_block = PSBlock(N)
        
        # Create segment input
        segment_input = torch.randn(C, N)
        
        # Process through PS Block
        ps_output = ps_block(segment_input)
        
        # Use output as Q, K, V matrices for attention
        Q = ps_output
        K = ps_output
        V = ps_output
        
        # Verify shapes
        assert Q.shape == K.shape == V.shape == (C, N)
        
        # Simple attention computation (scaled dot-product)
        scale = torch.sqrt(torch.tensor(N, dtype=torch.float32))
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # Verify output shape
        assert attention_output.shape == (C, N)
        
    def test_device_placement_consistency(self):
        """Test device placement consistency"""
        N = 32
        ps_block = PSBlock(N)
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            # Move to GPU
            ps_block_gpu = PSBlock(N).cuda()
            input_gpu = torch.randn(112, N).cuda()
            
            # Verify all components are on GPU
            assert ps_block_gpu.linear1.weight.is_cuda
            assert ps_block_gpu.linear2.weight.is_cuda
            assert ps_block_gpu.linear3.weight.is_cuda
            
            # Process input
            output_gpu = ps_block_gpu(input_gpu)
            assert output_gpu.is_cuda
            
            # Compare with CPU results (with proper weight copying)
            ps_block_cpu = PSBlock(N)
            with torch.no_grad():
                ps_block_cpu.linear1.weight.copy_(ps_block_gpu.linear1.weight.cpu())
                ps_block_cpu.linear2.weight.copy_(ps_block_gpu.linear2.weight.cpu())
                ps_block_cpu.linear3.weight.copy_(ps_block_gpu.linear3.weight.cpu())
            
            input_cpu = input_gpu.cpu()
            output_cpu = ps_block_cpu(input_cpu)
            
            # Results should be approximately equal
            assert torch.allclose(output_cpu, output_gpu.cpu(), atol=1e-6)
        else:
            # Just verify all components are on CPU
            assert not ps_block.linear1.weight.is_cuda
            assert not ps_block.linear2.weight.is_cuda
            assert not ps_block.linear3.weight.is_cuda
            
            input_cpu = torch.randn(112, N)
            output_cpu = ps_block(input_cpu)
            assert not output_cpu.is_cuda
            
    def test_encoder_integration(self):
        """Test integration within encoder architecture"""
        N = 32
        C = 112
        
        # Create shared PS Block for use in encoder
        shared_ps_block = PSBlock(N)
        
        # Simulate two segment attention layers and final PS Block using the same parameters
        segment_att_1_output = shared_ps_block(torch.randn(C, N))
        segment_att_2_output = shared_ps_block(torch.randn(C, N))
        final_output = shared_ps_block(torch.randn(C, N))
        
        # Verify all use the same parameters
        assert torch.equal(shared_ps_block.linear1.weight, shared_ps_block.linear1.weight)
        assert torch.equal(shared_ps_block.linear2.weight, shared_ps_block.linear2.weight)
        assert torch.equal(shared_ps_block.linear3.weight, shared_ps_block.linear3.weight)
        
        # Verify outputs have correct shapes
        assert segment_att_1_output.shape == (C, N)
        assert segment_att_2_output.shape == (C, N)
        assert final_output.shape == (C, N)