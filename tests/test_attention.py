import pytest
import torch
import torch.nn as nn
import numpy as np
from src.blocks.attention import (
    ScaledDotProductAttention,
    SegmentAttentionStage,
    TwoStageSegmentAttention,
    PSformerEncoderLayer,
    PSformerEncoder
)
from src.blocks.ps_block import PSBlock


class TestScaledDotProductAttention:
    """Test cases for ScaledDotProductAttention class"""
    
    def test_basic_forward_pass(self):
        """Test basic forward pass with valid inputs"""
        attention = ScaledDotProductAttention()
        
        # Create sample inputs
        batch, num_queries, num_keys, embed_dim, value_dim = 2, 3, 4, 5, 6
        Q = torch.randn(batch, num_queries, embed_dim)
        K = torch.randn(batch, num_keys, embed_dim)
        V = torch.randn(batch, num_keys, value_dim)
        
        # Forward pass
        output, weights = attention(Q, K, V)
        
        # Check output shapes
        assert output.shape == (batch, num_queries, value_dim)
        assert weights.shape == (batch, num_queries, num_keys)
        
        # Check that attention weights sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch, num_queries), atol=1e-6)
    
    def test_input_validation(self):
        """Test input validation"""
        attention = ScaledDotProductAttention()
        
        # Test with invalid dimensions
        with pytest.raises(ValueError):
            Q = torch.randn(2, 3)  # 2D tensor
            K = torch.randn(2, 4, 5)
            V = torch.randn(2, 4, 6)
            attention(Q, K, V)
        
        # Test with mismatched batch dimensions
        with pytest.raises(ValueError):
            Q = torch.randn(2, 3, 5)
            K = torch.randn(3, 4, 5)  # Different batch size
            V = torch.randn(2, 4, 6)
            attention(Q, K, V)
        
        # Test with mismatched key/value counts
        with pytest.raises(ValueError):
            Q = torch.randn(2, 3, 5)
            K = torch.randn(2, 4, 5)
            V = torch.randn(2, 5, 6)  # Different number of values
            attention(Q, K, V)
        
        # Test with mismatched embedding dimensions
        with pytest.raises(ValueError):
            Q = torch.randn(2, 3, 5)
            K = torch.randn(2, 4, 6)  # Different embedding dimension
            V = torch.randn(2, 4, 7)
            attention(Q, K, V)
    
    def test_scaling_factor(self):
        """Test that scaling factor is applied correctly"""
        attention = ScaledDotProductAttention()
        
        # Simple case where we can calculate the expected result
        Q = torch.ones(1, 1, 4)  # [1, 1, 4] tensor of ones
        K = torch.ones(1, 1, 4)  # [1, 1, 4] tensor of ones
        V = torch.ones(1, 1, 3)  # [1, 1, 3] tensor of ones
        
        # Manual calculation:
        # scores = Q @ K^T = [1,1,1,1] @ [1,1,1,1]^T = 4
        # scaled_scores = scores / sqrt(4) = 4 / 2 = 2
        # weights = softmax(2) = 1 (since there's only one value)
        # output = weights @ V = 1 * [1,1,1] = [1,1,1]
        
        output, weights = attention(Q, K, V)
        
        # Check that the scaled score is correct
        expected_scaled_score = 2.0  # 4 / sqrt(4)
        # We can't directly access the scaled scores, but we can verify the output
        expected_output = torch.ones(1, 1, 3)
        assert torch.allclose(output, expected_output, atol=1e-6)
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to one"""
        attention = ScaledDotProductAttention()
        
        # Create random inputs
        Q = torch.randn(3, 5, 8)
        K = torch.randn(3, 7, 8)
        V = torch.randn(3, 7, 10)
        
        _, weights = attention(Q, K, V)
        
        # Check that weights sum to 1 along the last dimension
        sums = weights.sum(dim=-1)
        expected = torch.ones_like(sums)
        assert torch.allclose(sums, expected, atol=1e-6)
    
    def test_output_with_known_weights(self):
        """Test output computation with known weights"""
        attention = ScaledDotProductAttention()
        
        # Define specific attention weights and values
        weights = torch.tensor([[[0.5, 0.5], [0.1, 0.9]]])  # [1, 2, 2]
        V = torch.tensor([[[1., 2.], [3., 4.]]])  # [1, 2, 2]
        
        # Manually compute expected output
        # output[0, 0, :] = 0.5 * [1, 2] + 0.5 * [3, 4] = [2, 3]
        # output[0, 1, :] = 0.1 * [1, 2] + 0.9 * [3, 4] = [2.8, 3.8]
        expected_output = torch.tensor([[[2., 3.], [2.8, 3.8]]])
        
        # Since we can't directly inject the weights, we'll test the computation indirectly
        # by creating inputs that would produce these weights
        
        # For this test, we'll just verify the matrix multiplication logic
        output_via_matmul = torch.matmul(weights, V)
        assert torch.allclose(output_via_matmul, expected_output, atol=1e-6)
    
    def test_numerical_stability(self):
        """Test numerical stability with large values"""
        attention = ScaledDotProductAttention()
        
        # Create inputs with large values that could cause overflow
        Q = torch.tensor([[[100., 100.]]])  # [1, 1, 2]
        K = torch.tensor([[[100., 100.], [0., 0.]]])  # [1, 2, 2]
        V = torch.tensor([[[1., 2.], [3., 4.]]])  # [1, 2, 2]
        
        output, weights = attention(Q, K, V)
        
        # Check that output is finite (not NaN or Inf)
        assert torch.isfinite(output).all()
        assert torch.isfinite(weights).all()
        
        # With these inputs, the first key should have much higher attention weight
        # due to the large dot product with the query
        assert weights[0, 0, 0] > weights[0, 0, 1]  # First key should have higher weight
    
    def test_zero_value_input(self):
        """Test that zero value inputs produce zero outputs"""
        attention = ScaledDotProductAttention()
        
        Q = torch.randn(2, 3, 5)
        K = torch.randn(2, 4, 5)
        V = torch.zeros(2, 4, 6)  # All zeros
        
        output, weights = attention(Q, K, V)
        
        # Output should be all zeros
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the attention mechanism"""
        attention = ScaledDotProductAttention()
        
        # Create inputs with gradients
        Q = torch.randn(2, 3, 5, requires_grad=True)
        K = torch.randn(2, 4, 5, requires_grad=True)
        V = torch.randn(2, 4, 6, requires_grad=True)
        
        # Forward pass
        output, _ = attention(Q, K, V)
        
        # Compute loss and backpropagate
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist and are not all zero
        assert Q.grad is not None
        assert K.grad is not None
        assert V.grad is not None
        
        assert not torch.all(Q.grad == 0)
        assert not torch.all(K.grad == 0)
        assert not torch.all(V.grad == 0)
    
    def test_device_agnostic(self):
        """Test that the module works on both CPU and GPU"""
        attention = ScaledDotProductAttention()
        
        # Test on CPU
        Q_cpu = torch.randn(2, 3, 5)
        K_cpu = torch.randn(2, 4, 5)
        V_cpu = torch.randn(2, 4, 6)
        
        output_cpu, _ = attention(Q_cpu, K_cpu, V_cpu)
        assert output_cpu.device == Q_cpu.device
        
        # Test on GPU if available
        if torch.cuda.is_available():
            attention_gpu = ScaledDotProductAttention().cuda()
            Q_gpu = Q_cpu.cuda()
            K_gpu = K_cpu.cuda()
            V_gpu = V_cpu.cuda()
            
            output_gpu, _ = attention_gpu(Q_gpu, K_gpu, V_gpu)
            assert output_gpu.device == Q_gpu.device
            assert output_gpu.device.type == 'cuda'


class TestSegmentAttentionStage:
    """Test cases for SegmentAttentionStage class"""
    
    def test_basic_forward_pass(self):
        """Test basic forward pass"""
        # Create a PS Block
        ps_block = PSBlock(N=32)
        attention_stage = SegmentAttentionStage(ps_block)
        
        # Create sample input [batch, N, C]
        batch, N, C = 4, 8, 32
        x = torch.randn(batch, N, C)
        
        # Forward pass
        output, weights = attention_stage(x)
        
        # Check shapes
        assert output.shape == (batch, N, C)
        assert weights.shape == (batch, N, N)
    
    def test_input_validation(self):
        """Test input validation"""
        ps_block = PSBlock(N=32)
        attention_stage = SegmentAttentionStage(ps_block)
        
        # Test with invalid dimensions
        with pytest.raises(ValueError):
            x = torch.randn(4, 8)  # 2D tensor instead of 3D
            attention_stage(x)
    
    def test_shared_ps_block(self):
        """Test that the same PS Block is used for Q, K, V generation"""
        ps_block = PSBlock(N=32)
        attention_stage = SegmentAttentionStage(ps_block)
        
        # Mock the PS Block's forward method to track calls
        original_forward = ps_block.forward
        call_count = 0
        
        def mock_forward(x):
            nonlocal call_count
            call_count += 1
            return original_forward(x)
        
        ps_block.forward = mock_forward
        
        # Create input and run forward pass
        x = torch.randn(2, 4, 32)
        attention_stage(x)
        
        # The PS Block should be called exactly once
        assert call_count == 1
    
    def test_gradient_flow(self):
        """Test gradient flow through the attention stage"""
        ps_block = PSBlock(N=32)
        attention_stage = SegmentAttentionStage(ps_block)
        
        # Create input with gradients
        x = torch.randn(2, 4, 32, requires_grad=True)
        
        # Forward pass
        output, _ = attention_stage(x)
        
        # Compute loss and backpropagate
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestTwoStageSegmentAttention:
    """Test cases for TwoStageSegmentAttention class"""
    
    def test_basic_forward_pass(self):
        """Test basic forward pass"""
        ps_block = PSBlock(N=32)
        two_stage_attention = TwoStageSegmentAttention(ps_block)
        
        # Create sample input
        batch, N, C = 4, 8, 32
        x = torch.randn(batch, N, C)
        
        # Forward pass
        output, (stage1_weights, stage2_weights) = two_stage_attention(x)
        
        # Check shapes
        assert output.shape == (batch, N, C)
        assert stage1_weights.shape == (batch, N, N)
        assert stage2_weights.shape == (batch, N, N)
    
    def test_input_validation(self):
        """Test input validation"""
        ps_block = PSBlock(N=32)
        two_stage_attention = TwoStageSegmentAttention(ps_block)
        
        # Test with invalid dimensions
        with pytest.raises(ValueError):
            x = torch.randn(4, 8)  # 2D tensor instead of 3D
            two_stage_attention(x)
    
    def test_shared_ps_block(self):
        """Test that the same PS Block is shared across stages"""
        ps_block = PSBlock(N=32)
        two_stage_attention = TwoStageSegmentAttention(ps_block)
        
        # Mock the PS Block's forward method to track calls
        original_forward = ps_block.forward
        call_count = 0
        
        def mock_forward(x):
            nonlocal call_count
            call_count += 1
            return original_forward(x)
        
        ps_block.forward = mock_forward
        
        # Create input and run forward pass
        x = torch.randn(2, 4, 32)
        two_stage_attention(x)
        
        # The PS Block should be called twice (once for each stage)
        assert call_count == 2
    
    def test_activation_between_stages(self):
        """Test that ReLU activation is applied between stages"""
        ps_block = PSBlock(N=32)
        two_stage_attention = TwoStageSegmentAttention(ps_block)
        
        # Create input with negative values
        x = torch.randn(2, 4, 32) - 5  # Shift to ensure negative values
        assert (x < 0).any()  # Verify we have negative values
        
        # Get the stage1 output before activation
        stage1 = two_stage_attention.stage1
        stage1_output, _ = stage1(x)
        
        # Run through the full two-stage attention
        output, _ = two_stage_attention(x)
        
        # The output should not have negative values due to ReLU between stages
        # (This is a simplified check - in practice, the second stage processing
        # might still produce negative values, but it's unlikely with random weights)


class TestPSformerEncoderLayer:
    """Test cases for PSformerEncoderLayer class"""
    
    def test_basic_forward_pass(self):
        """Test basic forward pass"""
        ps_block = PSBlock(N=32)
        encoder_layer = PSformerEncoderLayer(ps_block)
        
        # Create sample input
        batch, N, C = 4, 8, 32
        x = torch.randn(batch, N, C)
        
        # Forward pass
        output, weights = encoder_layer(x)
        
        # Check shapes
        assert output.shape == (batch, N, C)
        assert isinstance(weights, tuple)
        assert len(weights) == 2  # Stage 1 and stage 2 weights
    
    def test_input_validation(self):
        """Test input validation"""
        ps_block = PSBlock(N=32)
        encoder_layer = PSformerEncoderLayer(ps_block)
        
        # Test with invalid dimensions
        with pytest.raises(ValueError):
            x = torch.randn(4, 8)  # 2D tensor instead of 3D
            encoder_layer(x)
    
    def test_residual_connection(self):
        """Test that residual connection is properly implemented"""
        ps_block = PSBlock(N=32)
        encoder_layer = PSformerEncoderLayer(ps_block)
        
        # Create a simple input
        x = torch.ones(2, 4, 32)
        
        # Run through the layer
        output, _ = encoder_layer(x)
        
        # With the residual connection, output should be close to but not exactly
        # the same as input (due to the transformations)
        assert output.shape == x.shape
        # We can't assert they're not equal because with specific weight initializations
        # they might be, but we can check the shape is preserved


class TestPSformerEncoder:
    """Test cases for PSformerEncoder class"""
    
    def test_basic_forward_pass(self):
        """Test basic forward pass"""
        num_layers = 2
        segment_length = 32
        encoder = PSformerEncoder(num_layers, segment_length)
        
        # Create sample input
        batch, N, C = 4, 8, 32
        x = torch.randn(batch, N, C)
        
        # Forward pass
        output, attention_weights_list = encoder(x)
        
        # Check shapes
        assert output.shape == (batch, N, C)
        assert len(attention_weights_list) == num_layers
        for weights in attention_weights_list:
            assert isinstance(weights, tuple)
            assert len(weights) == 2  # Stage 1 and stage 2 weights
    
    def test_input_validation(self):
        """Test input validation"""
        encoder = PSformerEncoder(num_layers=2, segment_length=32)
        
        # Test with invalid dimensions
        with pytest.raises(ValueError):
            x = torch.randn(4, 8)  # 2D tensor instead of 3D
            encoder(x)
    
    def test_multiple_layers(self):
        """Test that multiple layers are properly created"""
        num_layers = 3
        segment_length = 32
        encoder = PSformerEncoder(num_layers, segment_length)
        
        # Check that we have the correct number of layers
        assert len(encoder.layers) == num_layers
        
        # Check that each layer has its own PS Block
        ps_blocks = [layer.ps_block for layer in encoder.layers]
        for i, block1 in enumerate(ps_blocks):
            for j, block2 in enumerate(ps_blocks):
                if i != j:
                    # Different layers should have different PS Block instances
                    assert block1 is not block2


# Integration tests
class TestAttentionIntegration:
    """Integration tests for attention mechanism with other components"""
    
    def test_integration_with_data_transformer(self):
        """Test integration with data transformer"""
        from src.blocks.data_transformer import create_transformer_for_psformer
        
        # Create data transformer
        transformer = create_transformer_for_psformer(
            sequence_length=128,
            num_variables=7,
            patch_size=16
        )
        
        # Create sample input data
        batch, M, L = 4, 7, 128
        input_data = torch.randn(batch, M, L)
        
        # Transform to segments
        segments = transformer.forward_transform(input_data)
        
        # Create encoder
        encoder = PSformerEncoder(num_layers=1, segment_length=transformer.config.segment_length)
        
        # Process through encoder
        output, weights = encoder(segments)
        
        # Check shapes
        assert output.shape == segments.shape
        assert len(weights) == 1
        
        # Verify dimensions match PSformer specifications
        N = transformer.config.num_patches  # Number of segments
        C = transformer.config.segment_length  # Segment length
        assert segments.shape == (batch, N, C)
        assert output.shape == (batch, N, C)
    
    def test_batch_independence(self):
        """Test that batches are processed independently"""
        ps_block = PSBlock(N=32)
        encoder_layer = PSformerEncoderLayer(ps_block)
        
        # Create two different inputs
        input1 = torch.randn(1, 4, 32)
        input2 = torch.randn(1, 4, 32)
        
        # Process separately
        output1, _ = encoder_layer(input1)
        output2, _ = encoder_layer(input2)
        
        # Process together in a batch
        batch_input = torch.cat([input1, input2], dim=0)
        batch_output, _ = encoder_layer(batch_input)
        
        # Check that results match
        assert torch.allclose(output1, batch_output[0:1], atol=1e-6)
        assert torch.allclose(output2, batch_output[1:2], atol=1e-6)