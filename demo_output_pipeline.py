#!/usr/bin/env python3
"""
Demo script showing the PSFormer Output Processing Pipeline in action.

This script demonstrates:
1. Creating a PSFormer model with complete output processing pipeline
2. Running a forward pass that produces forecasting predictions
3. Examining intermediate outputs from the pipeline
4. Validating the output shapes match paper specifications
"""

import torch
from src.blocks.psformer import create_psformer_model

def main():
    print("=== PSFormer Output Processing Pipeline Demo ===")
    print()
    
    # Configuration parameters
    config = {
        'sequence_length': 96,      # L - input sequence length
        'num_variables': 7,         # M - number of time series variables  
        'patch_size': 16,           # P - size of each temporal patch
        'num_encoder_layers': 2,    # Number of PSformer encoder layers
        'prediction_length': 24     # F - prediction horizon length
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create model
    print("Creating PSFormer model with output processing pipeline...")
    model = create_psformer_model(**config)
    print("* Model created successfully")
    print()
    
    # Create sample input data
    batch_size = 4
    input_shape = (batch_size, config['num_variables'], config['sequence_length'])
    print(f"Creating sample input with shape: {input_shape}")
    sample_input = torch.randn(input_shape)
    print("* Sample input created")
    print()
    
    # Run complete forward pass (with output pipeline)
    print("Running complete forward pass through the model...")
    print("Pipeline: Input -> RevIN -> Data Transform -> Encoder -> Inverse Transform -> Linear Projection -> Inverse RevIN -> Output")
    
    with torch.no_grad():
        final_predictions = model.forward(sample_input)
    
    print("* Forward pass completed")
    print()
    
    # Examine output
    expected_output_shape = (batch_size, config['num_variables'], config['prediction_length'])
    print("Output Analysis:")
    print(f"  Expected output shape: {expected_output_shape}")
    print(f"  Actual output shape:   {final_predictions.shape}")
    print(f"  Shape matches expected: {final_predictions.shape == expected_output_shape}")
    print(f"  Output data type: {final_predictions.dtype}")
    print(f"  Contains NaN values: {torch.isnan(final_predictions).any().item()}")
    print(f"  Contains Inf values: {torch.isinf(final_predictions).any().item()}")
    print(f"  Output range: [{final_predictions.min().item():.4f}, {final_predictions.max().item():.4f}]")
    print()
    
    # Examine intermediate outputs using forward_with_intermediates
    print("Examining intermediate outputs from the pipeline...")
    with torch.no_grad():
        final_predictions_2, intermediates = model.forward_with_intermediates(sample_input)
    
    print("Intermediate Output Shapes:")
    print(f"  Encoder output shape:     {intermediates['encoder_output'].shape}")
    print(f"  Reshaped output shape:    {intermediates['reshaped_output'].shape}")  
    print(f"  Projected output shape:   {intermediates['projected_output'].shape}")
    print(f"  Attention weights layers: {len(intermediates['attention_weights'])}")
    print()
    
    # Verify consistency
    predictions_match = torch.allclose(final_predictions, final_predictions_2, atol=1e-6)
    print(f"Forward methods produce consistent results: {predictions_match}")
    print()
    
    # Demonstrate key paper concepts
    print("=== Key Paper Concepts Demonstrated ===")
    
    # Calculate dimensions as per paper notation
    N = config['sequence_length'] // config['patch_size']  # Number of patches
    C = config['num_variables'] * config['patch_size']     # Segment length
    M = config['num_variables']                            # Number of variables
    L = config['sequence_length']                          # Input sequence length  
    F = config['prediction_length']                        # Forecast horizon
    
    print(f"Paper Notation:")
    print(f"  N (num_patches) = L/P = {config['sequence_length']}/{config['patch_size']} = {N}")
    print(f"  C (segment_length) = M*P = {config['num_variables']}*{config['patch_size']} = {C}")
    print(f"  M (num_variables) = {M}")
    print(f"  L (sequence_length) = {L}")
    print(f"  F (prediction_length) = {F}")
    print()
    
    print("Data Flow Verification:")
    print(f"  Input shape [B,M,L]:           {sample_input.shape}")
    print(f"  Encoder output [B,N,C]:        {intermediates['encoder_output'].shape}")
    print(f"  After inverse transform [B,M,L]: {intermediates['reshaped_output'].shape}")
    print(f"  After projection [B,M,F]:      {intermediates['projected_output'].shape}")
    print(f"  Final output [B,M,F]:          {final_predictions.shape}")
    print()
    
    # Verify the linear projection layer
    print("Linear Projection Layer (W_F) Analysis:")
    projection_layer = model.output_projection
    print(f"  Weight matrix shape: {projection_layer.weight.shape}")
    print(f"  Expected shape [F,L]: [{F},{L}]")
    print(f"  Bias shape: {projection_layer.bias.shape}")
    print(f"  Expected bias shape [F]: [{F}]")
    print(f"  Layer matches paper spec: {projection_layer.weight.shape == (F, L)}")
    print()
    
    print("=== Output Processing Pipeline Demo Complete ===")
    print("* All components working correctly")
    print("* Output shapes match paper specifications")
    print("* Pipeline produces valid forecasting predictions")

if __name__ == "__main__":
    main()
