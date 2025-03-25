"""
Train a 190M OLMo model on Wikipedia data.
This script:
1. Downloads a small subset of Wikipedia data
2. Tokenizes it using the OLMo tokenizer
3. Trains the OLMo 190M model on this data
4. Logs metrics and generates sample text during training
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
import socket
import math

from olmo_core.config import DType
from olmo_core.nn.transformer import TransformerConfig, InitMethod
from olmo_core.optim import AdamWConfig, OptimGroupOverride
from olmo_core.train import TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, GPUMemoryMonitorCallback, WandBCallback
from olmo_core.train.common import Duration
from olmo_core.train.trainer import Trainer
from olmo_core.data import NumpyDatasetConfig, NumpyDataLoaderConfig, TokenizerConfig, NumpyDatasetType
from olmo_core.utils import seed_all
from olmo_core.nn.functional import cross_entropy_loss

# === CONFIGURATION VARIABLES ===
# These match OLMo-core training configuration
TOTAL_STEPS = 1000
BATCH_SIZE = 64  # Local batch size per GPU (scaled to match global batch size)
CHECKPOINT_INTERVAL = 500
INFERENCE_INTERVAL = 50
INFERENCE_PROMPT = "Dutch is "
LEARNING_RATE = 4e-4
SEQUENCE_LENGTH = 1024
Z_LOSS_MULTIPLIER = 1e-5  # Match OLMo-core configuration (in scripts/train/OLMo2-190M.py)
WANDB_API_KEY = "8e07447fa3d6269331f7ecd0b27f8518c2a65855"

# Constants for managing batch sizes
GLOBAL_BATCH_SIZE = BATCH_SIZE * SEQUENCE_LENGTH  # 4 * 1024 = 4096 tokens
MICRO_BATCH_SIZE = BATCH_SIZE * SEQUENCE_LENGTH  # 4 * 1024 = 4096 tokens (smaller than global batch size)

# Calculate the exact number of tokens needed for training
# Each step needs GLOBAL_BATCH_SIZE tokens
TOTAL_TOKENS_NEEDED = TOTAL_STEPS * GLOBAL_BATCH_SIZE
# Add 20% more tokens as a safety margin
TOTAL_TOKENS_WITH_MARGIN = int(TOTAL_TOKENS_NEEDED * 1.2)
# Calculate number of sequences needed
NUM_SEQUENCES_NEEDED = (TOTAL_TOKENS_NEEDED + SEQUENCE_LENGTH - 1) // SEQUENCE_LENGTH

def download_and_tokenize_wiki_subset(output_dir, tokenizer_config):
    """Download Wikipedia data, tokenize it to get exactly the needed number of tokens, and save as .npy file."""
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
            
            # Check if we have enough sequences
            total_tokens = tokens.shape[0] * tokens.shape[1]
            log_message(f"Existing data has {total_tokens} tokens")
            log_message(f"Need {TOTAL_TOKENS_NEEDED} tokens for training ({NUM_SEQUENCES_NEEDED} sequences)")
            
            if total_tokens >= TOTAL_TOKENS_NEEDED and tokens.shape[1] == SEQUENCE_LENGTH:
                log_message(f"Existing data is sufficient for training")
                return output_file
            else:
                log_message(f"Existing data is insufficient or has incorrect sequence length: {tokens.shape[1]}, expected {SEQUENCE_LENGTH}")
        except Exception as e:
            log_message(f"Error loading existing file: {e}")
            log_message("Will recreate the tokenized data")
    
    # Download Wikipedia dataset
    log_message("Downloading Wikipedia dataset...")
    log_message(f"Need to collect at least {TOTAL_TOKENS_NEEDED} tokens for training")
    log_message(f"Will download approximately {TOTAL_TOKENS_WITH_MARGIN} tokens including margin")
    
    # Load Wikipedia data
    wiki_dataset = load_dataset(
        "wikipedia", 
        "20220301.en", 
        split="train", 
        cache_dir=os.environ["HF_DATASETS_CACHE"]
    )
    
    log_message(f"Downloaded full Wikipedia dataset with {len(wiki_dataset)} articles")
    
    # Import the tokenizer - specifically using GPT NeoX OLMo Dolma v1.5 
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
    
    # Tokenize the data
    log_message("Tokenizing Wikipedia articles...")
    all_token_ids = []
    processed_articles = 0
    
    # Randomly shuffle article indices for diverse data
    total_articles = len(wiki_dataset)
    article_indices = np.random.permutation(total_articles)
    
    for idx in tqdm(article_indices):
        if processed_articles % 10 == 0:  # Log periodically
            log_message(f"Tokenizing article {processed_articles}, collected {len(all_token_ids)} tokens so far")
            if len(all_token_ids) > 0:
                log_message(f"Progress: {len(all_token_ids) / TOTAL_TOKENS_WITH_MARGIN * 100:.2f}% of target")
        
        article = wiki_dataset[int(idx)]
        text = article["text"]
        
        # Tokenize the text
        tokens = tokenizer.encode(text)
        
        # Explicitly filter out token ID 0
        tokens = [t for t in tokens if t != 0]
        
        # Also ensure tokens are within vocabulary range
        if len(tokens) > 0 and max(tokens) >= vocab_size:
            # log_message(f"Warning: Article {processed_articles} contains tokens outside vocabulary range")
            tokens = [t for t in tokens if t < vocab_size]
        
        all_token_ids.extend(tokens)
        processed_articles += 1
        
        # Break early if we have enough tokens
        if len(all_token_ids) >= TOTAL_TOKENS_WITH_MARGIN:
            log_message(f"Collected sufficient tokens: {len(all_token_ids)}")
            break
    
    log_message(f"Tokenization complete. Processed {processed_articles} articles.")
    log_message(f"Total tokens collected: {len(all_token_ids)}")
    
    # Log maximum token ID for diagnostics
    if all_token_ids:
        max_token = max(all_token_ids)
        log_message(f"Maximum token ID in dataset: {max_token}")
        
        # Verify token ID 0 has been filtered out
        token_0_count = all_token_ids.count(0)
        log_message(f"Token ID 0 count in dataset after filtering: {token_0_count}")
        if token_0_count > 0:
            log_message("WARNING: Token ID 0 still exists in dataset despite filtering - will perform second pass filtering")
            all_token_ids = [t for t in all_token_ids if t != 0]
            log_message(f"After second filtering: Token ID 0 count = {all_token_ids.count(0)}")
    else:
        raise ValueError("No tokens were generated from the Wikipedia articles.")
    
    # Ensure we have enough tokens for training
    if len(all_token_ids) < TOTAL_TOKENS_NEEDED:
        # If we don't have enough tokens, repeat what we have
        tokens_needed = TOTAL_TOKENS_NEEDED - len(all_token_ids)
        repetitions = (tokens_needed // len(all_token_ids)) + 1
        log_message(f"Don't have enough tokens. Need {tokens_needed} more tokens.")
        log_message(f"Repeating existing tokens {repetitions} times to reach target")
        all_token_ids = all_token_ids * (repetitions + 1)
        log_message(f"After repetition: {len(all_token_ids)} tokens")
    
    # Reshape into sequences
    num_sequences = len(all_token_ids) // SEQUENCE_LENGTH
    token_sequences = np.array(all_token_ids[:num_sequences * SEQUENCE_LENGTH]).reshape(-1, SEQUENCE_LENGTH)
    
    log_message(f"Created {token_sequences.shape[0]} sequences of {SEQUENCE_LENGTH} tokens")
    log_message(f"Total number of tokens: {token_sequences.shape[0] * SEQUENCE_LENGTH}")
    
    # Verify we have enough for training
    have_tokens = token_sequences.shape[0] * SEQUENCE_LENGTH
    log_message(f"Need {TOTAL_TOKENS_NEEDED} tokens for training, have {have_tokens} tokens")
    
    if have_tokens < TOTAL_TOKENS_NEEDED:
        log_message(f"WARNING: Still don't have enough tokens after repetition")
    
    # Save as .npy file
    file_path = os.path.join(output_dir, "wiki_tokens.npy")
    log_message(f"Saving {token_sequences.shape[0]} sequences of {SEQUENCE_LENGTH} tokens to {file_path}")
    np.save(file_path, token_sequences.astype(np.int32))
    log_message(f"Successfully saved tokenized data to {file_path}")
    
    return file_path

def run_inference(model, tokenizer_config, inference_tokenizer, step, device="cuda"):
    """Generate text with the model for monitoring training progress."""
    log_message(f"Running inference at step {step}...")
    
    # Save original model state
    was_training = model.training
    model.eval()  # Set to evaluation mode
    
    # Tokenize the prompt
    tokens = inference_tokenizer.encode(INFERENCE_PROMPT)
    
    # Ensure no token ID 0 in prompt
    tokens = [t for t in tokens if t != 0]
    
    token_tensor = torch.tensor([tokens], device=device)
    
    with torch.no_grad():
        # Initial forward pass
        logits = model(token_tensor)  # OLMo returns logits directly
        
        # Setup generation
        generated_tokens = tokens.copy()
        max_new_tokens = 50
        temperature = 0.8  # Add some randomness
        
        for _ in range(max_new_tokens):
            # Get predictions for next token
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
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
    generated_text = inference_tokenizer.decode(generated_tokens)
    
    # Log output
    message = f"INFERENCE at Step {step}: \"{generated_text}\""
    log_message(message)
    
    # Log to W&B
    if wandb.run is not None:
        wandb.log({
            "inference/generated_text": wandb.Html(f"<p>{generated_text}</p>"),
            "inference/generated_length": len(generated_tokens) - len(tokens)
        }, step=step)
    
    # Restore original model state
    if was_training:
        model.train()
    
    return generated_text

# Custom cross-entropy loss function matching OLMo-core implementation
def compute_loss(logits, labels, ignore_index=-100, z_loss_multiplier=Z_LOSS_MULTIPLIER):
    """
    Compute cross-entropy loss with z-loss, matching OLMo-core implementation.
    """
    loss, z_loss = cross_entropy_loss(
        logits=logits,
        labels=labels,
        ignore_index=ignore_index,
        compute_z_loss=True,
        z_loss_multiplier=z_loss_multiplier
    )
    
    # In OLMo-core, z_loss is returned separately but included in the total loss
    total_loss = loss
    if z_loss is not None:
        total_loss = loss + z_loss
    
    return total_loss, {"ce_loss": loss, "z_loss": z_loss if z_loss is not None else 0.0}

if __name__ == "__main__":
    try:
        start_time = time.time()
        print(f"Script started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Set up Weights & Biases for detailed logging
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        
        # Set CUDA_LAUNCH_BLOCKING for better error messages
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
        # Set specific GPU device - use GPU 1 which is less utilized
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use only GPU 1
        log_message(f"Setting CUDA_VISIBLE_DEVICES to use GPU 1")
        
        # Dynamically set data directory based on hostname
        hostname = socket.gethostname()
        if 'vibranium' in hostname.lower():
            data_dir = "/data/s4422090"
        elif 'alice' in hostname.lower() or 'nodelogin' in hostname.lower():
            data_dir = "/data1/s4422090"
        else:
            # Default fallback
            data_dir = os.path.join(os.getcwd(), "data")
        
        # Set up cache directories
        cache_dir = os.path.join(data_dir, "huggingface_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # Set all cache-related environment variables
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
        os.environ["TORCH_HOME"] = os.path.join(data_dir, "torch")
        os.environ["XDG_CACHE_HOME"] = os.path.join(data_dir, "xdg_cache")
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_dir, "hub")
        os.environ["WANDB_DIR"] = os.path.join(data_dir, "wandb")
        
        # Create all cache directories
        for cache_path in [os.environ["HF_DATASETS_CACHE"], 
                          os.environ["TRANSFORMERS_CACHE"], 
                          os.environ["TORCH_HOME"],
                          os.environ["XDG_CACHE_HOME"],
                          os.environ["HUGGINGFACE_HUB_CACHE"]]:
            os.makedirs(cache_path, exist_ok=True)
        
        # Set up logging to both console and file
        log_file = os.path.join(data_dir, "olmo_training.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        def log_message(message):
            """Log a message to both console and file."""
            print(message, flush=True)
            with open(log_file, "a") as f:
                f.write(f"{message}\n")
        
        # Set random seed for reproducibility
        seed_all(42)
        log_message("Set random seed to 42")
        
        # Get device - explicitly use CUDA device 0 (which is physically GPU 1 due to CUDA_VISIBLE_DEVICES)
        device = torch.device("cuda:0")
        log_message(f"Using device: {device} (physically GPU 1)")
        
        # Make sure CUDA is available
        if not torch.cuda.is_available():
            log_message("CUDA is not available! Falling back to CPU.")
            device = torch.device("cpu")
        else:
            # Log GPU information
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
            log_message(f"Using GPU: {gpu_name} with {gpu_memory:.2f} GB memory")
            
            # Set default device
            torch.cuda.set_device(0)  # Set default CUDA device
        
        # Configure tokenizer - using the exact OLMo-core tokenizer
        tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
        log_message(f"Configured tokenizer: gpt_neox_olmo_dolma_v1_5 with vocab size {tokenizer_config.vocab_size}")
        log_message(f"Padded vocab size: {tokenizer_config.padded_vocab_size()}")
        
        # Setup for inference
        from transformers import AutoTokenizer
        inference_tokenizer = AutoTokenizer.from_pretrained(
            "allenai/gpt-neox-olmo-dolma-v1_5",
            cache_dir=os.path.join(os.environ["TRANSFORMERS_CACHE"], "tokenizers")
        )
        
        # Download and tokenize Wikipedia data - now based on exact token count
        wiki_data_dir = os.path.join(data_dir, "wiki_data")
        log_message(f"Starting download and tokenization to collect {TOTAL_TOKENS_NEEDED} tokens...")
        wiki_data_path = download_and_tokenize_wiki_subset(
            wiki_data_dir, 
            tokenizer_config
        )
        log_message(f"Completed download and tokenization. Data saved to {wiki_data_path}")
        
        # Configure model with exact OLMo2-190M configuration matching OLMo-core
        log_message("Configuring model with OLMo-core settings...")
        log_message("Token ID 0 will be excluded from all operations")
        model_config = TransformerConfig.olmo2_190M(
            vocab_size=tokenizer_config.padded_vocab_size(),
            compile=False,  # Disable compilation for simplicity on a single GPU
            dtype=DType.bfloat16,
            init_method=InitMethod.normal,  # Explicit normal initialization as in OLMo-core
        )
        log_message(f"Model configured with vocab size {tokenizer_config.padded_vocab_size()}")
        log_message(f"Using initialization method: {InitMethod.normal}")
        
        # Build model directly on the device
        log_message("Building model...")
        model = model_config.build(
            init_device=device,
            device=device,
            max_seq_len=SEQUENCE_LENGTH,
            mesh=None,  # No distributed mesh for single GPU
        )
        log_message("Model built successfully")

        # Special handling for token ID 0 - set embedding to zeros and logit bias to large negative
        log_message("Applying special handling for token ID 0...")
        with torch.no_grad():
            # Set embedding for token ID 0 to zeros (virtually removing it from the model's knowledge)
            model.embeddings.weight[0].zero_()
            
            # Set output layer bias for token ID 0 to large negative value
            # This ensures it won't be predicted during inference
            if hasattr(model.lm_head, 'w_out') and model.lm_head.w_out.bias is not None:
                model.lm_head.w_out.bias[0] = -100.0
        log_message("Special handling for token ID 0 applied")

        # Check for NaN values in model parameters
        log_message("Checking model parameters for NaN values...")
        has_nan = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                log_message(f"NaN found in parameter: {name}")
                has_nan = True
        if not has_nan:
            log_message("No NaN values found in model parameters")
        
        # Log the embedding weight initialization pattern
        log_message("Checking embedding weight statistics...")
        embed_weight = model.embeddings.weight
        log_message(f"Embedding weight shape: {embed_weight.shape}")
        log_message(f"Embedding weight mean: {embed_weight.mean().item()}")
        log_message(f"Embedding weight std: {embed_weight.std().item()}")
        log_message(f"Embedding weight min: {embed_weight.min().item()}")
        log_message(f"Embedding weight max: {embed_weight.max().item()}")
        
        # Configure optimizer exactly matching OLMo2-190M configuration from OLMo-core
        log_message("Configuring optimizer with OLMo-core settings...")
        optim_config = AdamWConfig(
            lr=LEARNING_RATE,
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
        
        # Set up paths for training data and checkpoints
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        save_folder = os.path.join(data_dir, f"olmo_wiki_training_output_{timestamp}")
        os.makedirs(save_folder, exist_ok=True)
        log_message(f"Created save folder: {save_folder}")
        
        # Initialize wandb
        wandb.init(
            project="olmo-190m-wikipedia",
            name=f"olmo-190m-wiki-{timestamp}",
            config={
                "model": "OLMo-190M",
                "batch_size": BATCH_SIZE,
                "total_steps": TOTAL_STEPS,
                "sequence_length": SEQUENCE_LENGTH,
                "learning_rate": LEARNING_RATE,
                "z_loss_multiplier": Z_LOSS_MULTIPLIER,
                "inference_interval": INFERENCE_INTERVAL,
                "checkpoint_interval": CHECKPOINT_INTERVAL,
                "initialization_method": "normal",
                "weight_decay": 0.1,
                "optim_betas": (0.9, 0.95),
            }
        )
        
        # Configure dataset
        log_message("Setting up dataset...")
        dataset_config = NumpyDatasetConfig(
            tokenizer=tokenizer_config,
            name=NumpyDatasetType.fsl,
            paths=[wiki_data_path],
            sequence_length=SEQUENCE_LENGTH,
            work_dir=os.path.join(save_folder, "dataset_work_dir"),
        )
        
        # Build dataset
        log_message("Building dataset...")
        dataset = dataset_config.build()
        log_message(f"Dataset built successfully with {len(dataset)} samples")
        
        # Configure data loader - try to match OLMo's global batch size
        log_message("Configuring data loader...")
        data_loader_config = NumpyDataLoaderConfig(
            global_batch_size=GLOBAL_BATCH_SIZE,  # Use the defined constant
            seed=42,
            num_workers=1,
        )
        
        # Build data loader
        log_message("Building data loader...")
        data_loader = data_loader_config.build(dataset)
        log_message("Data loader built successfully")
        
        # Configure trainer with OLMo-core settings
        log_message("Configuring trainer with OLMo-core settings...")
        trainer_config = TrainerConfig(
            save_folder=save_folder,
            rank_microbatch_size=MICRO_BATCH_SIZE,  # Use smaller micro-batch size
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=1,  # Match OLMo-core
            z_loss_multiplier=Z_LOSS_MULTIPLIER,  # Match OLMo-core z-loss value
            compile_loss=False,  # Disable compilation for simplicity
            max_duration=Duration.steps(TOTAL_STEPS),
            device=str(device),
            load_path=None,  # Explicitly disable checkpoint loading
        ).with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=CHECKPOINT_INTERVAL,
                ephemeral_save_interval=CHECKPOINT_INTERVAL // 2,
                save_async=False,  # Disable async saving for simplicity
            ),
        ).with_callback(
            "gpu_monitor",
            GPUMemoryMonitorCallback(),
        ).with_callback(
            "wandb",
            WandBCallback(
                name=wandb.run.name if wandb.run else "olmo-190m-wiki",
                entity=None,  # Use default entity
                project="olmo-190m-wikipedia",
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
        
        # Monkey patch the trainer to perform inference at intervals
        log_message("Setting up inference during training...")
        original_train_batch = trainer._train_batch
        
        # Define a new method that wraps the original
        def patched_train_batch(self, batch, dry_run=False):
            # Call the original method first
            result = original_train_batch(batch, dry_run=dry_run)
            
            # Check if we should do inference
            if (not dry_run and 
                self.global_step > 0 and 
                self.global_step % INFERENCE_INTERVAL == 0):
                log_message(f"Triggering inference at step {self.global_step}")
                run_inference(self.model, tokenizer_config, inference_tokenizer, self.global_step, device=device)
                
            return result
        
        # Apply the patch
        trainer._train_batch = patched_train_batch.__get__(trainer)
        log_message("Patched trainer._train_batch to include inference")
        
        # Run initial inference with the untrained model
        log_message("Running inference with untrained model...")
        run_inference(model, tokenizer_config, inference_tokenizer, 0, device=device)
        
        # Start training
        log_message("Starting training...")
        trainer.fit()
        log_message("Training complete!")
        
        # Run final inference
        log_message("Running final inference with trained model...")
        run_inference(model, tokenizer_config, inference_tokenizer, TOTAL_STEPS, device=device)
        
        # Close wandb
        wandb.finish()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        log_message(f"Script completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log_message(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    except Exception as e:
        log_message(f"ERROR: {e}")
        log_message(traceback.format_exc())
        # Log to wandb if active
        if wandb.run is not None:
            wandb.run.finish(exit_code=1)
        raise
