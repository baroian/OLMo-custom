"""
Train a 190M OLMo model on a small subset (0.01%) of Wikipedia data.
This script:
1. Downloads a small subset of Wikipedia data
2. Tokenizes it using the OLMo tokenizer
3. Converts it to the .npy format expected by OLMo
4. Trains the model on this data
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import wandb
from datasets import load_dataset
from tqdm import tqdm
import time
import traceback
import shutil

from olmo_core.config import DType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, OptimGroupOverride
from olmo_core.train import TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, GPUMemoryMonitorCallback, WandBCallback, Callback
from olmo_core.train.common import Duration
from olmo_core.train.trainer import Trainer
from olmo_core.data import NumpyDatasetConfig, NumpyDataLoaderConfig, TokenizerConfig, NumpyDatasetType
from olmo_core.utils import seed_all



#### EDIT THE SCALE OF TRAINING HERE 
#### scale_factor = 1 means 1000 steps, 
#### scale_factor controls 1) the number of steps, 2) the number of tokens, 3) the number of samples
#### batch_size_factor = 8 means 1024 * 8 tokens per batch
#### total tokens = 1024 * scale_factor * batch_size_factor

scale_factor = 0.1
batch_size_factor = 4



# Set CUDA_LAUNCH_BLOCKING for better error messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Set up directories to use /data instead of home directory

# Dynamically set data directory based on hostname
hostname = socket.gethostname()
if 'vibranium' in hostname.lower():
    data_dir = "/data/s4422090"
elif 'alice' in hostname.lower() or 'nodelogin' in hostname.lower():
    data_dir = "/data1/s4422090"
else:
    # Default fallback
    data_dir = "/data1/s4422090"
    print(f"Unknown hostname: {hostname}, defaulting to {data_dir}")
    
cache_dir = os.path.join(data_dir, "huggingface_cache")
os.makedirs(cache_dir, exist_ok=True)

# Set all cache-related environment variables to use /data/s4422090
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
os.environ["TORCH_HOME"] = os.path.join(data_dir, "torch")
os.environ["XDG_CACHE_HOME"] = os.path.join(data_dir, "xdg_cache")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_dir, "hub")
os.environ["WANDB_DIR"] = os.path.join(data_dir, "wandb")

# Create all cache directories explicitly
for cache_path in [os.environ["HF_DATASETS_CACHE"], 
                  os.environ["TRANSFORMERS_CACHE"], 
                  os.environ["TORCH_HOME"],
                  os.environ["XDG_CACHE_HOME"],
                  os.environ["HUGGINGFACE_HUB_CACHE"]]:
    os.makedirs(cache_path, exist_ok=True)

# Set up logging to both console and file
log_file = os.path.join(data_dir, "wiki_training.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)

def log_message(message):
    """Log a message to both console and file."""
    print(message, flush=True)
    with open(log_file, "a") as f:
        f.write(f"{message}\n")

log_message(f"Script started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
log_message(f"Using cache directory: {cache_dir}")
log_message(f"Environment variables set:")
for env_var in ["HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME", "HUGGINGFACE_HUB_CACHE", "WANDB_DIR"]:
    log_message(f"  {env_var}={os.environ.get(env_var, 'Not set')}")

def download_and_tokenize_wiki_subset(output_dir, tokenizer_config, num_samples=200):
    """Download a small subset of Wikipedia data, tokenize it, and save as .npy file."""
    os.makedirs(output_dir, exist_ok=True)
    log_message(f"Created output directory: {output_dir}")
    
    # Check if the tokenized file already exists
    output_file = os.path.join(output_dir, "wiki_tokens.npy")
    if os.path.exists(output_file):
        log_message(f"Found existing tokenized data at {output_file}")
        try:
            # Verify the file is valid
            tokens = np.load(output_file)
            log_message(f"Loaded existing tokenized data with shape {tokens.shape}")
            if tokens.shape[1] == 1024:  # Check if the sequence length is correct
                return output_file
            else:
                log_message(f"Existing data has incorrect sequence length: {tokens.shape[1]}, expected 1024")
        except Exception as e:
            log_message(f"Error loading existing file: {e}")
            log_message("Will recreate the tokenized data")
    
    # Download Wikipedia dataset
    log_message("Downloading Wikipedia dataset (tiny subset)...")
    # Load a very small subset of Wikipedia data
    wiki_dataset = load_dataset(
        "wikipedia", 
        "20220301.en", 
        split="train", 
        cache_dir=os.environ["HF_DATASETS_CACHE"]
    )
    
    # Take a small portion of the dataset (0.2%)
    total_size = len(wiki_dataset)
    subset_size = max(int(total_size * 0.002 * scale_factor), 1)  # 0.2% of total, minimum 1 sample
    
    log_message(f"Downloaded full Wikipedia dataset with {total_size} articles")
    log_message(f"Taking {subset_size} articles ({scale_factor * 0.2}% of the dataset)")
    
    # Select a random subset
    indices = np.random.choice(total_size, size=subset_size, replace=False)
    wiki_subset = wiki_dataset.select(indices)
    
    log_message(f"Selected {len(wiki_subset)} Wikipedia articles")
    
    # Take only a small subset of what we downloaded
    subset_size = min(num_samples, len(wiki_subset))
    if subset_size < len(wiki_subset):
        wiki_subset = wiki_subset.select(range(subset_size))
        log_message(f"Further reduced to {subset_size} Wikipedia articles")
    
    # Import the tokenizer - specifically using GPT NeoX OLMo Dolma v1.5 with no fallback
    log_message("Loading GPT NeoX OLMo Dolma v1.5 tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/gpt-neox-olmo-dolma-v1_5",
        cache_dir=os.path.join(os.environ["TRANSFORMERS_CACHE"], "tokenizers")
    )
    log_message("Successfully loaded gpt-neox-olmo-dolma-v1_5 tokenizer")
    
    # Determine the actual vocab size of the loaded tokenizer
    actual_vocab_size = len(tokenizer)
    vocab_size = tokenizer_config.padded_vocab_size()
    log_message(f"Actual tokenizer vocabulary size: {actual_vocab_size}")
    log_message(f"OLMo padded vocabulary size: {vocab_size}")
    
    if actual_vocab_size > vocab_size:
        log_message(f"WARNING: Tokenizer vocabulary size ({actual_vocab_size}) exceeds model vocabulary size ({vocab_size})")
        log_message("Will ensure all token IDs are capped to model vocabulary size")
    
    # Tokenize the data
    log_message("Tokenizing Wikipedia articles...")
    all_token_ids = []
    
    for i, article in enumerate(tqdm(wiki_subset)):
        if i % 5 == 0:  # Log even more frequently
            log_message(f"Tokenizing article {i}/{len(wiki_subset)}")
        
        text = article["text"]
        # Tokenize the text
        tokens = tokenizer.encode(text)
        
        # Ensure all token IDs are strictly within the model's vocabulary range
        max_allowed_id = min(vocab_size - 100, 50000)  # Even safer margin
        safe_tokens = [min(t, max_allowed_id) for t in tokens]
        all_token_ids.extend(safe_tokens)
        
        # Periodically check the maximum token ID to catch issues early
        if i % 10 == 0 and all_token_ids:
            max_token = max(all_token_ids) if all_token_ids else 0
            if max_token >= max_allowed_id:
                log_message(f"WARNING: Found token ID {max_token} >= safe limit {max_allowed_id} at article {i}")
                # Cap all token IDs to the safe limit
                all_token_ids = [min(t, max_allowed_id) for t in all_token_ids]
                log_message(f"Capped all token IDs to maximum of {max_allowed_id}")
    
    log_message(f"Tokenization complete. Total tokens: {len(all_token_ids)}")
    
    # Final safety check on token IDs
    if all_token_ids:
        max_token = max(all_token_ids)
        log_message(f"Maximum token ID in dataset: {max_token}")
        if max_token >= max_allowed_id:
            log_message(f"WARNING: Found token ID {max_token} >= safe limit {max_allowed_id}")
            all_token_ids = [min(t, max_allowed_id) for t in all_token_ids]
            log_message(f"Capped all token IDs to maximum of {max_allowed_id}")
    else:
        raise ValueError("No tokens were generated from the Wikipedia articles.")
    
    # Ensure we have enough tokens (at least 1024 per sample)
    sequence_length = 1024
    min_tokens_needed = sequence_length * num_samples
    if len(all_token_ids) < min_tokens_needed:
        # Repeat the tokens if necessary
        repetitions = (min_tokens_needed // len(all_token_ids)) + 1
        all_token_ids = all_token_ids * repetitions
        log_message(f"Repeated tokens {repetitions} times to ensure enough data")
    
    # Reshape into sequences of 1024 tokens
    num_sequences = len(all_token_ids) // sequence_length
    token_sequences = np.array(all_token_ids[:num_sequences * sequence_length]).reshape(-1, sequence_length)
    
    # Save as .npy file
    file_path = os.path.join(output_dir, "wiki_tokens.npy")
    log_message(f"Saving {token_sequences.shape[0]} sequences of {sequence_length} tokens to {file_path}")
    np.save(file_path, token_sequences.astype(np.int32))
    log_message(f"Successfully saved tokenized data to {file_path}")
    
    return file_path

def main():
    # Set up Weights & Biases
    wandb_api_key = "8e07447fa3d6269331f7ecd0b27f8518c2a65855"
    os.environ["WANDB_API_KEY"] = wandb_api_key
    log_message("Set up Weights & Biases API key")
    
    # Set up paths - use /data directory to avoid quota issues
    save_folder = os.path.join(data_dir, "olmo_wiki_subset_output")
    
    # Delete any existing checkpoints to avoid dataset mismatch errors
    if os.path.exists(save_folder):
        log_message(f"Removing existing checkpoint directory: {save_folder}")
        shutil.rmtree(save_folder)
    
    os.makedirs(save_folder, exist_ok=True)
    log_message(f"Created save folder: {save_folder}")
    
    # Set random seed
    seed_all(42)
    log_message("Set random seed to 42")
    
    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {device}")
    
    # Configure tokenizer - using GPT NeoX OLMo Dolma v1.5 instead of Dolma2
    tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()  # Changed from Dolma2 to GPT NeoX OLMo Dolma v1.5
    log_message(f"Configured tokenizer: GPT NeoX OLMo Dolma v1.5 with vocab size {tokenizer_config.vocab_size}")
    log_message(f"Padded vocab size: {tokenizer_config.padded_vocab_size()}")
    
    # Setup for inference
    from transformers import AutoTokenizer
    inference_tokenizer = AutoTokenizer.from_pretrained(
        "allenai/gpt-neox-olmo-dolma-v1_5",
        cache_dir=os.path.join(os.environ["TRANSFORMERS_CACHE"], "tokenizers")
    )
    
    # Set up inference
    inference_prompt = "Amsterdam is "
    inference_file = os.path.join(data_dir, "inference_outputs.txt")
    with open(inference_file, "w") as f:
        f.write(f"Inference outputs starting at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Prompt: \"{inference_prompt}\"\n\n")
    
    log_message(f"Set up inference with prompt: \"{inference_prompt}\"")
    log_message(f"Inference outputs will be saved to: {inference_file}")
    
    # Function to run inference
    def run_inference(model, step):
        try:
            log_message(f"Running inference at step {step}...")
            
            # Save original model state
            was_training = model.training
            model.eval()  # Set to evaluation mode
            
            # Tokenize the prompt
            tokens = inference_tokenizer.encode(inference_prompt)
            token_tensor = torch.tensor([tokens], device=device)
            
            with torch.no_grad():
                # Initial forward pass - OLMo returns tensor directly, not an object with logits
                logits = model(token_tensor)  # This is the logits tensor directly
                
                # Greedy decoding
                generated_tokens = tokens.copy()
                max_new_tokens = 50
                
                for _ in range(max_new_tokens):
                    # Get predictions for next token
                    next_token_logits = logits[0, -1, :]
                    next_token = torch.argmax(next_token_logits).item()
                    
                    # Stop if EOS token
                    if next_token == tokenizer_config.eos_token_id:
                        break
                    
                    # Add token and continue
                    generated_tokens.append(next_token)
                    next_input = torch.tensor([generated_tokens], device=device)
                    logits = model(next_input)  # Again, this is the logits tensor directly
            
            # Decode the generated text
            generated_text = inference_tokenizer.decode(generated_tokens)
            
            # Log output
            message = f"INFERENCE at Step {step}: \"{generated_text}\""
            log_message(message)
            
            # Save to dedicated file
            with open(inference_file, "a") as f:
                f.write(f"Step {step}: {generated_text}\n\n")
            
            # Log to W&B if available
            if wandb.run is not None:
                wandb.log({
                    "inference/generated_text": wandb.Html(f"<p>{generated_text}</p>"),
                    "inference/generated_length": len(generated_tokens) - len(tokens)
                }, step=step)
            
            # Restore original model state
            if was_training:
                model.train()
                
            log_message(f"Inference at step {step} completed successfully")
            return generated_text
            
        except Exception as e:
            log_message(f"Error during inference at step {step}: {e}")
            log_message(traceback.format_exc())
            return None
    
    # Download and tokenize Wikipedia data
    wiki_data_dir = os.path.join(data_dir, "wiki_data")
    sequence_length = 1024
    num_samples = int(2000 * scale_factor)  # Ensure the number of samples is an integer
    log_message(f"Starting download and tokenization of {num_samples} Wikipedia articles...")
    wiki_data_path = download_and_tokenize_wiki_subset(
        wiki_data_dir, 
        tokenizer_config,
        num_samples
    )
    log_message(f"Completed download and tokenization. Data saved to {wiki_data_path}")
    
    # Configure model with simpler settings
    log_message("Configuring model...")
    model_config = TransformerConfig.olmo2_190M(
        vocab_size=tokenizer_config.padded_vocab_size(),
        compile=False,  # Disable compilation for simplicity
        dp_config=None,  # Disable distributed training
    )
    log_message(f"Model configured with vocab size {tokenizer_config.padded_vocab_size()}")
    
    # Build model directly on the device
    log_message("Building model...")
    model = model_config.build(
        init_device=device,
        device=device,
        max_seq_len=sequence_length,
        mesh=None,  # No distributed mesh
    )
    log_message("Model built successfully")

    # Check for NaN values in model parameters
    log_message("Checking model parameters for NaN values...")
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            log_message(f"NaN found in parameter: {name}")
            has_nan = True
    if not has_nan:
        log_message("No NaN values found in model parameters")
    
    # Configure optimizer
    log_message("Configuring optimizer...")
    optim_config = AdamWConfig(
        lr=1e-4,  # Reduced learning rate for better stability
        weight_decay=0.1,
        betas=(0.9, 0.95),
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
        fused=True,
    )
    
    # Build optimizer
    log_message("Building optimizer...")
    optimizer = optim_config.build(model)
    log_message("Optimizer built successfully")
    
    # Configure dataset using the Wikipedia data directly
    log_message("Setting up Wikipedia dataset...")
    
    # Create a dataset config that uses the Wikipedia data
    dataset_config = NumpyDatasetConfig(
        tokenizer=tokenizer_config,
        name=NumpyDatasetType.fsl,
        paths=[wiki_data_path],
        sequence_length=sequence_length,
        work_dir=os.path.join(save_folder, "dataset_work_dir"),
    )
    
    # Build dataset
    log_message("Building dataset...")
    dataset = dataset_config.build()
    log_message(f"Dataset built successfully with {len(dataset)} samples")
    
    # Configure data loader with smaller batch size
    log_message("Configuring data loader...")
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size= 1024 * batch_size_factor,  # Keep batch size small for stability
        seed=42,
        num_workers=1,
    )
    
    # Build data loader
    log_message("Building data loader...")
    data_loader = data_loader_config.build(dataset)
    log_message("Data loader built successfully")
    
    # Run name for this experiment
    run_name = "olmo_190m_wiki_subset_gpt_neox_v1_5"  # Updated to reflect the use of GPT NeoX OLMo Dolma v1.5 tokenizer
    
    # Configure trainer
    log_message("Configuring trainer...")
    trainer_config = TrainerConfig(
        save_folder=save_folder,
        rank_microbatch_size=1024,  # Microbatch size must match sequence length
        save_overwrite=True,
        metrics_collect_interval=10,  # Increased from 1 to 10 for efficiency with longer training
        cancel_check_interval=10,  # Increased from 1 to 10 for efficiency
        z_loss_multiplier=1e-3,  # Increased z_loss for better stability
        compile_loss=False,  # Disable compilation for simplicity
        # Set a limited number of steps for the test run
        max_duration=Duration.steps(int(1000 * scale_factor)),  # Convert to int to avoid float error
        device=str(device),  # Explicitly set device
    ).with_callback(
        "checkpointer",
        CheckpointerCallback(
            save_interval=100,  # Save checkpoint every 100 steps
            ephemeral_save_interval=50,  # Save ephemeral checkpoint every 50 steps
            save_async=False,  # Disable async saving for simplicity
        ),
    ).with_callback(
        "gpu_monitor",
        GPUMemoryMonitorCallback(),
    ).with_callback(
        "wandb",
        WandBCallback(
            name=run_name,
            entity="s4422090-universiteit-leiden",  # Your wandb team name
            project="olmo-190m-wiki-subset",
            enabled=True,
            cancel_check_interval=10,
        ),
    )
    log_message("Trainer configured successfully")
    
    # Build trainer
    log_message("Building trainer...")
    trainer = trainer_config.build(
        model=model, 
        optim=optimizer, 
        data_loader=data_loader,
        mesh=None,  # No distributed mesh
    )
    log_message("Trainer built successfully")
    
    # -------------------------------
    # Monkey patch the trainer to perform inference at intervals
    # -------------------------------
    
    # Store the original train_batch method
    original_train_batch = trainer._train_batch
    inference_interval = 50
    
    # Define a new method that wraps the original
    def patched_train_batch(self, batch, dry_run=False):
        # Call the original method first
        result = original_train_batch(batch, dry_run=dry_run)
        
        # Check if we should do inference
        if (not dry_run and 
            self.global_step > 0 and 
            self.global_step % inference_interval == 0):
            log_message(f"Triggering inference at step {self.global_step}")
            run_inference(self.model, self.global_step)
            
        return result
    
    # Apply the patch
    trainer._train_batch = patched_train_batch.__get__(trainer)
    log_message("Patched trainer._train_batch to include inference")
    
    # Run initial inference with the untrained model
    log_message("Running inference with untrained model...")
    run_inference(model, 0)
    
    # Start training
    log_message("Starting training...")
    trainer.fit()
    log_message("Training complete!")
    
    # Run final inference
    log_message("Running final inference with trained model...")
    run_inference(model, int(1000 * scale_factor))

if __name__ == "__main__":
    main() 