###### NOT clean, but easy implementation rather than making it modular

"""
### USE THIS IN MAIN TO APPLY LAYER-WISE SCALING

# Configure model with simpler settings
log_message("Configuring model...")

# Get the standard 190M model configuration
model_config = TransformerConfig.olmo2_190M(
    vocab_size=tokenizer_config.padded_vocab_size(),
    compile=False,  # Disable compilation for simplicity
    dp_config=None,  # Disable distributed training
)

# Apply layer-wise scaling
log_message("Applying layer-wise scaling to model...")
model, scaling_info = apply_layer_wise_scaling(
    model_config,
    qkv_multipliers=(0.5, 1.0),  # Start with 0.5x heads, end with 1.0x
    ffn_multipliers=(0.5, 4.0)   # Start with 0.5x FFN width, end with 4.0x
)

# Log layer-wise scaling information
log_message(f"Layer-wise scaling applied with parameters:")
log_message(f"  QKV multipliers: {scaling_info['qkv_multipliers']}")
log_message(f"  FFN multipliers: {scaling_info['ffn_multipliers']}")

log_message("\nFeed-forward hidden size per layer:")
for i, size in enumerate(scaling_info['ffn_sizes_per_layer']):
    log_message(f"  Layer {i}: {size} hidden size ({scaling_info['ffn_multipliers_per_layer'][i]}x multiplier)")

# Move model to the target device
model = model.to(device)
log_message("Model transferred to device")

"""


def apply_layer_wise_scaling(model_config, qkv_multipliers=(0.5, 1.0), ffn_multipliers=(0.5, 4.0)):
    """
    Applies layer-wise scaling to an OLMo model configuration.
    
    Args:
        model_config: A TransformerConfig object (typically from TransformerConfig.olmo2_190M())
        qkv_multipliers: Tuple of (min, max) multipliers for attention dimensions
        ffn_multipliers: Tuple of (min, max) multipliers for feed-forward dimensions
        
    Returns:
        model: A built and modified model with layer-wise scaling applied
    """
    import numpy as np
    import torch
    from olmo_core.nn.feed_forward import FeedForward
    
    # Extract key dimensions from the config
    d_model = model_config.d_model
    n_layers = model_config.n_layers
    
    # Build the model first
    model = model_config.build(
        init_device="cpu",  # First build on CPU for modification
    )
    
    # Calculate the linearly spaced multipliers for each layer
    ffn_multipliers_per_layer = [
        round(float(x), 2) 
        for x in np.linspace(
            ffn_multipliers[0],
            ffn_multipliers[1],
            num=n_layers,
            dtype=float
        )
    ]
    
    qkv_multipliers_per_layer = [
        round(float(x), 2)
        for x in np.linspace(
            qkv_multipliers[0],
            qkv_multipliers[1],
            num=n_layers,
            dtype=float
        )
    ]
    
    # Store the scaled dimensions for logging
    ffn_sizes_per_layer = []
    
    # Modify each layer with appropriate scaling
    for i in range(n_layers):
        # Get the original block
        original_block = model.blocks[i]
        
        # Apply FFN scaling to this layer
        ffn_multiplier = ffn_multipliers_per_layer[i]
        ffn_hidden_size = int(round(d_model * ffn_multiplier))
        
        # Make divisible by 256 (common in OLMo)
        hidden_size_multiple_of = 256
        ffn_hidden_size = hidden_size_multiple_of * (
            (ffn_hidden_size + hidden_size_multiple_of - 1) // hidden_size_multiple_of
        )
        
        # Store for later reference
        ffn_sizes_per_layer.append(ffn_hidden_size)
        
        # Get the old feed forward module
        old_ffn = original_block.feed_forward
        
        # Create a new feed forward module with scaled hidden size
        new_ffn = FeedForward(
            d_model=d_model,
            hidden_size=ffn_hidden_size,
            bias=False if not hasattr(old_ffn.w1, 'bias') or old_ffn.w1.bias is None else True,
            dtype=model_config.dtype.as_pt(),
            init_device="cpu"
        )
        
        # Transfer weights with appropriate scaling
        with torch.no_grad():
            # For w1, we take the first n columns or expand as needed
            if old_ffn.w1.weight.shape[0] > new_ffn.w1.weight.shape[0]:
                new_ffn.w1.weight.copy_(old_ffn.w1.weight[:new_ffn.w1.weight.shape[0], :])
            else:
                # If we need more rows, repeat and add noise
                repeat_factor = new_ffn.w1.weight.shape[0] // old_ffn.w1.weight.shape[0] + 1
                repeated = old_ffn.w1.weight.repeat(repeat_factor, 1)
                new_ffn.w1.weight.copy_(repeated[:new_ffn.w1.weight.shape[0], :])
                # Add small noise to break symmetry
                new_ffn.w1.weight.add_(torch.randn_like(new_ffn.w1.weight) * 0.01)
            
            # Similar for w3 (SwiGLU gate) if it exists
            if hasattr(old_ffn, 'w3') and hasattr(new_ffn, 'w3'):
                if old_ffn.w3.weight.shape[0] > new_ffn.w3.weight.shape[0]:
                    new_ffn.w3.weight.copy_(old_ffn.w3.weight[:new_ffn.w3.weight.shape[0], :])
                else:
                    repeat_factor = new_ffn.w3.weight.shape[0] // old_ffn.w3.weight.shape[0] + 1
                    repeated = old_ffn.w3.weight.repeat(repeat_factor, 1)
                    new_ffn.w3.weight.copy_(repeated[:new_ffn.w3.weight.shape[0], :])
                    new_ffn.w3.weight.add_(torch.randn_like(new_ffn.w3.weight) * 0.01)
            
            # For w2, we take the first n rows or expand as needed
            if old_ffn.w2.weight.shape[1] > new_ffn.w2.weight.shape[1]:
                new_ffn.w2.weight.copy_(old_ffn.w2.weight[:, :new_ffn.w2.weight.shape[1]])
            else:
                # If we need more columns, repeat and add noise
                repeat_factor = new_ffn.w2.weight.shape[1] // old_ffn.w2.weight.shape[1] + 1
                repeated = old_ffn.w2.weight.repeat(1, repeat_factor)
                new_ffn.w2.weight.copy_(repeated[:, :new_ffn.w2.weight.shape[1]])
                new_ffn.w2.weight.add_(torch.randn_like(new_ffn.w2.weight) * 0.01)
        
        # Replace the feed forward module in the block
        original_block.feed_forward = new_ffn
    
    # Store the scaling parameters as attributes for reference
    model.ffn_multipliers = ffn_multipliers
    model.qkv_multipliers = qkv_multipliers
    model.ffn_multipliers_per_layer = ffn_multipliers_per_layer
    model.qkv_multipliers_per_layer = qkv_multipliers_per_layer
    model.ffn_sizes_per_layer = ffn_sizes_per_layer
    
    return model, {
        "ffn_multipliers": ffn_multipliers,
        "qkv_multipliers": qkv_multipliers,
        "ffn_multipliers_per_layer": ffn_multipliers_per_layer,
        "qkv_multipliers_per_layer": qkv_multipliers_per_layer,
        "ffn_sizes_per_layer": ffn_sizes_per_layer
    }