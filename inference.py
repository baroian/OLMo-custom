"""
OLMo Model Inference Script

This script loads a trained OLMo model from a checkpoint and runs inference to generate text.
"""

import os
import torch
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm

from olmo_core.config import DType
from olmo_core.nn.transformer import TransformerConfig, InitMethod, TransformerActivationCheckpointingMode
from olmo_core.train.train_module.transformer.config import TransformerActivationCheckpointingConfig
from olmo_core.data import TokenizerConfig
from olmo_core.utils import seed_all

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run inference with a trained OLMo model")
parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file (e.g., data/olmo_wiki_training_output_XXX/stepXXXX/train/rank0.pt)")
parser.add_argument("--prompt", type=str, default="Dutch is ", help="Prompt for inference")
parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of tokens to generate (default: 100)")
parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature (default: 0.8)")
parser.add_argument("--seed", type=int, default=421, help="Random seed for reproducibility (default: 42)")
args = parser.parse_args()

# Set up device
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)

# Set random seed
seed_all(args.seed)

# Set up cache directory
data_dir = f"/home/{os.environ.get('USER')}/OLMo-custom/data"
cache_dir = os.path.join(data_dir, "huggingface_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

# Configure tokenizer
tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
print(f"Configured tokenizer with vocab size {tokenizer_config.padded_vocab_size()}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "allenai/gpt-neox-olmo-dolma-v1_5",
    cache_dir=os.path.join(os.environ["TRANSFORMERS_CACHE"], "tokenizers")
)

# Set up model configuration
model_config = TransformerConfig.olmo2_190M(
    vocab_size=tokenizer_config.padded_vocab_size(),
    dtype=DType.bfloat16,
    init_method=InitMethod.normal
)

# Set activation checkpointing
ac_config = TransformerActivationCheckpointingConfig(
    mode=TransformerActivationCheckpointingMode.full
)
model_config.activation_checkpointing = ac_config

# Build model
print("Building model...")
model = model_config.build(init_device=device)

# Load checkpoint
print(f"Loading checkpoint from {args.checkpoint}...")
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

# --- DEBUG: Print checkpoint keys ---
print("Checkpoint keys:", checkpoint.keys() if isinstance(checkpoint, dict) else "Not a dict")
# --- END DEBUG ---

# If the checkpoint contains the model in a specific format, extract it
if isinstance(checkpoint, dict) and "model" in checkpoint:
    model_state = checkpoint["model"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model_state = checkpoint["state_dict"]
elif isinstance(checkpoint, dict):
    # Fallback if keys are different but it's still a dictionary
    print("Warning: Checkpoint dictionary does not contain 'model' or 'state_dict'. Attempting to load entire dictionary as state.")
    model_state = checkpoint
else:
    # If checkpoint is not a dictionary (e.g., just the state_dict itself)
    print("Warning: Checkpoint does not appear to be a dictionary. Attempting to load directly.")
    model_state = checkpoint

# Load state into model
print("Attempting to load state dictionary...")
# Let's try strict=True first to see if it reveals missing/unexpected keys
try:
    model.load_state_dict(model_state, strict=True)
    print("Successfully loaded state dictionary (strict=True).")
except RuntimeError as e:
    print(f"Strict loading failed: {e}")
    print("Attempting to load state dictionary with strict=False...")
    model.load_state_dict(model_state, strict=False)
    print("Successfully loaded state dictionary (strict=False).")

print("Model loaded successfully")

# Set model to evaluation mode
model.eval()

# Special handling for token ID 0 - for safety
with torch.no_grad():
    # Set embedding for token ID 0 to zeros
    model.embeddings.weight[0].zero_()
    
    # Set output layer bias for token ID 0 to large negative value
    if hasattr(model.lm_head, 'w_out') and model.lm_head.w_out.bias is not None:
        model.lm_head.w_out.bias[0] = -100.0

print("\n" + "="*50)
print(f"Running inference with prompt: '{args.prompt}'")
print("="*50 + "\n")

# Tokenize the prompt
tokens = tokenizer.encode(args.prompt)

# Ensure no token ID 0 in prompt
tokens = [t for t in tokens if t != 0]

# Convert tokens to tensor and move to device
token_tensor = torch.tensor([tokens], device=device)

# Generate text
with torch.no_grad():
    # Initial forward pass
    logits = model(token_tensor)
    
    # Setup generation
    generated_tokens = tokens.copy()
    
    # Generation loop
    for _ in tqdm(range(args.max_tokens), desc="Generating tokens"):
        # Get predictions for next token
        next_token_logits = logits[0, -1, :]
        
        # Apply temperature
        next_token_logits = next_token_logits / args.temperature
        
        # Create a mask to filter out token ID 0
        mask = torch.ones_like(next_token_logits, dtype=torch.bool)
        mask[0] = False  # Mask out token ID 0
        
        # Apply the mask (set masked logits to negative infinity)
        masked_logits = next_token_logits.clone()
        masked_logits[~mask] = -float('inf')
        
        # Apply softmax to masked logits
        probs = torch.nn.functional.softmax(masked_logits, dim=-1)
        
        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        # Stop if EOS token
        if next_token == tokenizer_config.eos_token_id:
            break
        
        # Add token and continue
        generated_tokens.append(next_token)
        next_input = torch.tensor([generated_tokens], device=device)
        logits = model(next_input)

# Decode the generated text
generated_text = tokenizer.decode(generated_tokens)

# Print the result
print("\n" + "="*50)
print("Generated text:")
print("-"*50)
print(generated_text)
print("="*50)

# Print some stats
print(f"Generated {len(generated_tokens) - len(tokens)} new tokens")
print(f"Total sequence length: {len(generated_tokens)}")
