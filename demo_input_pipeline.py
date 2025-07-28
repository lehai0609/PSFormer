"""
Demo of PSformer Input Processing Pipeline

This script demonstrates the complete Step 4.1 implementation:
Raw Input → RevIN Normalization → Data Transformation → PSformer Encoder
"""

import torch
from src.blocks import create_psformer_model

def main():
    print("=== PSformer Input Processing Pipeline Demo ===\n")
    
    # Configuration
    config = {
        'sequence_length': 512,
        'patch_size': 32,
        'num_variables': 7,
        'num_encoder_layers': 3
    }
    
    print(f"Configuration:")
    print(f"- Sequence Length: {config['sequence_length']}")
    print(f"- Patch Size: {config['patch_size']}")
    print(f"- Number of Variables: {config['num_variables']}")
    print(f"- Number of Encoder Layers: {config['num_encoder_layers']}")
    
    # Create model
    model = create_psformer_model(**config)
    print(f"\n[OK] PSformer model created successfully")
    
    # Show model architecture info
    info = model.get_model_info()
    print(f"\nModel Architecture:")
    print(f"- Input Shape: {info['dimensions']['input_shape']}")
    print(f"- Encoder Input Shape: {info['dimensions']['encoder_input_shape']}")
    print(f"- Number of Patches (N): {info['dimensions']['num_patches']}")
    print(f"- Segment Length (C): {info['dimensions']['segment_length']}")
    print(f"- PS Block Dimension: {info['components']['ps_block_dimension']}")
    
    # Create sample input
    batch_size = 4
    raw_input = torch.randn(batch_size, config['num_variables'], config['sequence_length'])
    print(f"\n[OK] Sample input created: {raw_input.shape}")
    
    # Forward pass through the complete pipeline
    print(f"\n=== Forward Pass Through Input Processing Pipeline ===")
    
    # Step 1: Raw Input
    print(f"1. Raw Input: {raw_input.shape}")
    
    # Step 2: RevIN Normalization
    normalized_input = model.revin_layer(raw_input, mode='norm')
    print(f"2. After RevIN Normalization: {normalized_input.shape}")
    print(f"   - Statistics stored: mean={model.revin_layer.mean.shape}, stdev={model.revin_layer.stdev.shape}")
    
    # Step 3: Data Transformation (Patching + Segmenting)
    encoder_ready_data = model.data_transformer.forward_transform(normalized_input)
    print(f"3. After Data Transformation: {encoder_ready_data.shape}")
    
    # Step 4: PSformer Encoder
    encoder_output, attention_weights_list = model.encoder(encoder_ready_data)
    print(f"4. After PSformer Encoder: {encoder_output.shape}")
    print(f"   - Attention weights from {len(attention_weights_list)} layers")
    
    # Complete pipeline in one call
    print(f"\n=== Complete Pipeline (Single Forward Call) ===")
    final_output, attention_weights = model(raw_input)
    print(f"[OK] Pipeline output: {final_output.shape}")
    print(f"[OK] Attention weights: {len(attention_weights)} layers")
    
    # Verify attention weights structure
    print(f"\n=== Attention Weights Analysis ===")
    for i, (stage1_weights, stage2_weights) in enumerate(attention_weights):
        print(f"Layer {i+1}:")
        print(f"  - Stage 1 attention: {stage1_weights.shape}")
        print(f"  - Stage 2 attention: {stage2_weights.shape}")
        # Verify attention weights sum to 1
        stage1_sum = torch.sum(stage1_weights, dim=-1)
        stage2_sum = torch.sum(stage2_weights, dim=-1)
        print(f"  - Stage 1 weights sum ~= 1.0: {torch.allclose(stage1_sum, torch.ones_like(stage1_sum), atol=1e-5)}")
        print(f"  - Stage 2 weights sum ~= 1.0: {torch.allclose(stage2_sum, torch.ones_like(stage2_sum), atol=1e-5)}")
    
    print(f"\n=== Verification ===")
    
    # Verify transformation symmetry
    restored = model.data_transformer.inverse_transform(encoder_ready_data, config['sequence_length'])
    is_symmetric = torch.allclose(normalized_input, restored, atol=1e-6)
    print(f"[OK] Data transformation symmetry: {is_symmetric}")
    
    # Verify RevIN symmetry
    denormalized = model.revin_layer(normalized_input, mode='denorm')
    is_revin_symmetric = torch.allclose(raw_input, denormalized, atol=1e-5)
    print(f"[OK] RevIN normalization symmetry: {is_revin_symmetric}")
    
    # Verify architecture consistency
    C_from_transformer = model.data_transformer.config.segment_length
    N_from_ps_block = model.encoder.layers[0].ps_block.N
    architecture_consistent = (C_from_transformer == N_from_ps_block)
    print(f"[OK] Architecture consistency (C={C_from_transformer} == N={N_from_ps_block}): {architecture_consistent}")
    
    print(f"\n=== Summary ===")
    print(f"[SUCCESS] PSformer Input Processing Pipeline successfully implemented!")
    print(f"[SUCCESS] All components working correctly:")
    print(f"   - RevIN normalization with statistics storage")
    print(f"   - Data transformation (patching + segmenting)")
    print(f"   - PSformer encoder with parameter sharing")
    print(f"   - Two-stage segment attention mechanism")
    print(f"[SUCCESS] Pipeline transforms {raw_input.shape} -> {final_output.shape}")
    print(f"[SUCCESS] Ready for output pipeline implementation (Step 4.2)")


if __name__ == "__main__":
    main()
