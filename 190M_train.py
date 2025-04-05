"""
Train a 190M OLMo model on Wikipedia data.
This script:
1. Downloads a small subset of Wikipedia data
2. Tokenizes it using the OLMo tokenizer
3. Trains the OLMo 190M model on this data
4. Logs metrics and generates sample text during training
"""

### Make a Weights & Biases account (so i can add to the team) and put your API key here
WANDB_API_KEY = "8e07447fa3d6269331f7ecd0b27f8518c2a65855"


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import wandb
from datasets import load_dataset
from tqdm import tqdm
import time
import argparse
import numpy as np
from transformers import AutoTokenizer

from olmo_core.config import DType
from olmo_core.nn.transformer import TransformerConfig, InitMethod, TransformerActivationCheckpointingMode
from olmo_core.optim import AdamWConfig, OptimGroupOverride
from olmo_core.train import TrainerConfig
from olmo_core.train.common import Duration
from olmo_core.train.trainer import Trainer
from olmo_core.data import NumpyDatasetConfig, NumpyDataLoaderConfig, TokenizerConfig, NumpyDatasetType
from olmo_core.utils import seed_all
from olmo_core.train.train_module.transformer.config import TransformerActivationCheckpointingConfig, TransformerTrainModuleConfig
from olmo_core.train.train_module.transformer import TransformerTrainModule
from olmo_core.train.callbacks import WandBCallback, Callback

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train a 190M OLMo model on Wikipedia data")
parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")
parser.add_argument("--steps", type=int, default=100, help="Total training steps (default: 100)")
parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
parser.add_argument("--prompt", type=str, default="Dutch is ", help="Prompt for inference")
parser.add_argument("--data-dir", type=str, default=None, help="Data directory to use")
parser.add_argument("--inference-interval", type=int, default=200, 
                   help="Run inference every N steps (default: 200)")
parser.add_argument("--inference-prompt", type=str, default="Dutch is ", 
                   help="Prompt to use for inference (default: 'Dutch is ')")
args = parser.parse_args()

# Constants
SEQUENCE_LENGTH = 1024
GLOBAL_BATCH_SIZE = args.batch_size * SEQUENCE_LENGTH
TOTAL_STEPS = args.steps

# Calculate exactly how much data we need
TOTAL_TOKENS_NEEDED = TOTAL_STEPS * GLOBAL_BATCH_SIZE
TOTAL_TOKENS_WITH_MARGIN = int(TOTAL_TOKENS_NEEDED * 1.1)  # 10% margin
SEQUENCES_NEEDED = (TOTAL_TOKENS_WITH_MARGIN + SEQUENCE_LENGTH - 1) // SEQUENCE_LENGTH

print(f"Training parameters:")
print(f"- Batch size: {args.batch_size} sequences")
print(f"- Training steps: {TOTAL_STEPS}")
print(f"- Sequence length: {SEQUENCE_LENGTH}")
print(f"- Total tokens needed: {TOTAL_TOKENS_NEEDED:,}")
print(f"- With 10% margin: {TOTAL_TOKENS_WITH_MARGIN:,}")
print(f"- Sequences needed: {SEQUENCES_NEEDED:,}")

# Set up data directories
data_dir = args.data_dir or f"/home/{os.environ.get('USER')}/OLMo-custom/data"
os.makedirs(data_dir, exist_ok=True)
cache_dir = os.path.join(data_dir, "huggingface_cache")
os.makedirs(cache_dir, exist_ok=True)

# Download and tokenize data
wiki_data_dir = os.path.join(data_dir, "wiki_data")
os.makedirs(wiki_data_dir, exist_ok=True)
wiki_data_path = os.path.join(wiki_data_dir, "wiki_tokens.npy")

# Always delete existing data file to force re-download
if os.path.exists(wiki_data_path):
    print("Removing existing data file to force re-download")
    os.remove(wiki_data_path)

# Set environment variables for cache directories
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
os.environ["HF_HOME"] = os.path.join(cache_dir, "huggingface")
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Set device
device = torch.device(f"cuda:{args.gpu}")
torch.cuda.set_device(args.gpu)

# Set random seed
seed_all(42)

# Configure tokenizer
tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
print(f"Configured tokenizer with vocab size {tokenizer_config.padded_vocab_size()}")

# Download and tokenize data
if not os.path.exists(wiki_data_path):
    # Download Wikipedia dataset
    print("Data file not found, downloading and tokenizing Wikipedia data")
    
    # Load tokenizer for tokenization
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/gpt-neox-olmo-dolma-v1_5",
        cache_dir=os.path.join(os.environ["TRANSFORMERS_CACHE"], "tokenizers")
    )
    
    # Download Wikipedia dataset
    print("Downloading Wikipedia dataset...")
    wiki_dataset = load_dataset(
        "wikipedia", 
        "20220301.en", 
        split="train", 
        cache_dir=os.environ["HF_DATASETS_CACHE"]
    )
    print(f"Downloaded full Wikipedia dataset with {len(wiki_dataset)} articles")
    
    # Tokenize articles until we have enough tokens
    all_tokens = []
    articles_processed = 0
    print(f"Tokenizing articles until we collect {TOTAL_TOKENS_WITH_MARGIN} tokens...")

    # Get a subset of articles to process with progress bar
    max_articles_to_try = min(100000, len(wiki_dataset))
    for article in tqdm(wiki_dataset.select(range(max_articles_to_try)), desc="Processing articles"):
        # Tokenize article text
        tokens = tokenizer.encode(article["text"])
        # Filter out token ID 0 (if any)
        tokens = [t for t in tokens if t != 0]
        all_tokens.extend(tokens)
        
        articles_processed += 1
        if articles_processed % 1000 == 0:
            print(f"Processed {articles_processed} articles, collected {len(all_tokens)} tokens so far")
            print(f"Progress: {len(all_tokens) / TOTAL_TOKENS_WITH_MARGIN * 100:.2f}% of target")
        
        if len(all_tokens) >= TOTAL_TOKENS_WITH_MARGIN:
            break
    
    print(f"Collected {len(all_tokens)} tokens from {articles_processed} articles")
    
    # Ensure we have enough tokens
    if len(all_tokens) < TOTAL_TOKENS_WITH_MARGIN:
        print(f"Warning: Could only collect {len(all_tokens)} tokens, which is less than the {TOTAL_TOKENS_WITH_MARGIN} requested")
    
    # Reshape tokens into sequences
    num_sequences = len(all_tokens) // SEQUENCE_LENGTH
    if num_sequences < SEQUENCES_NEEDED:
        print(f"Warning: Only able to create {num_sequences} sequences, which is less than the {SEQUENCES_NEEDED} needed")
    
    # Take only complete sequences
    tokens_to_use = all_tokens[:num_sequences * SEQUENCE_LENGTH]
    sequences = np.array(tokens_to_use, dtype=np.int32).reshape(-1, SEQUENCE_LENGTH)
    
    print(f"Created {sequences.shape[0]} sequences of length {SEQUENCE_LENGTH}")
    
    # Save tokenized data
    np.save(wiki_data_path, sequences)
    print(f"Saved tokenized data to {wiki_data_path}")

else:
    # Check if existing data is sufficient
    data = np.load(wiki_data_path)
    print(f"Loaded existing data with shape: {data.shape}")
    
    if data.shape[0] < SEQUENCES_NEEDED:
        print(f"Warning: Existing data has only {data.shape[0]} sequences, but {SEQUENCES_NEEDED} are needed")
        print("Consider re-running with more data by deleting the existing data file")
    
    # Check for token ID 0 in the data
    if np.any(data == 0):
        print("Warning: Token ID 0 found in the data. Replacing with token ID 1...")
        data = data.astype(np.int32)  # Ensure correct dtype
        data[data == 0] = 1
        np.save(wiki_data_path, data)
        print(f"Saved cleaned data to {wiki_data_path}")

# Load the data for training
print(f"Loading data from {wiki_data_path}")
wiki_tokens = np.load(wiki_data_path)

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
model = model_config.build(init_device=device)
print("Model built successfully")

# Special handling for token ID 0 - set embedding to zeros and logit bias to large negative
with torch.no_grad():
    # Set embedding for token ID 0 to zeros (virtually removing it from the model's knowledge)
    model.embeddings.weight[0].zero_()
    
    # Set output layer bias for token ID 0 to large negative value
    # This ensures it won't be predicted during inference
    if hasattr(model.lm_head, 'w_out') and model.lm_head.w_out.bias is not None:
        model.lm_head.w_out.bias[0] = -100.0
print("Special handling for token ID 0 applied - zeroed embeddings and set negative bias")

# Configure optimizer
optim_config = AdamWConfig(
    lr=4e-4,
    weight_decay=0.1,
    betas=(0.9, 0.95),
    group_overrides=[
        OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
    ],
    fused=True,
)

# Configure dataset
dataset_config = NumpyDatasetConfig(
    tokenizer=tokenizer_config,
    name=NumpyDatasetType.fsl,
    paths=[wiki_data_path],
    sequence_length=SEQUENCE_LENGTH,
    work_dir=os.path.join(data_dir, "dataset_work_dir"),
)

# Build dataset
dataset = dataset_config.build()
print(f"Dataset built with {len(dataset)} samples")

# Configure data loader
data_loader_config = NumpyDataLoaderConfig(
    global_batch_size=GLOBAL_BATCH_SIZE,
    seed=42,
    num_workers=4,
)

# Build data loader
data_loader = data_loader_config.build(dataset)
print("Data loader built successfully")

# Create save folder for checkpoints
timestamp = time.strftime('%Y%m%d-%H%M%S')
save_folder = os.path.join(data_dir, f"olmo_wiki_training_output_{timestamp}")
os.makedirs(save_folder, exist_ok=True)

# Create TransformerTrainModuleConfig
train_module_config = TransformerTrainModuleConfig(
    rank_microbatch_size=GLOBAL_BATCH_SIZE,
    max_sequence_length=SEQUENCE_LENGTH,
    optim=optim_config,
    compile_model=False,  # Set to False for compatibility
)

# Build train module from the config
train_module = train_module_config.build(model=model, device=device)
print("Train module built successfully")

# Set up Weights & Biases for detailed logging
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

# Initialize wandb
wandb.init(
    project="olmo-190m-wikipedia",
    name=f"190m-train-{timestamp}",
    config={
        "model": "OLMo-190M",
        "batch_size": args.batch_size,
        "total_steps": TOTAL_STEPS,
        "sequence_length": SEQUENCE_LENGTH,
        "learning_rate": 4e-4,
        "inference_interval": 50,
        "initialization_method": "normal",
        "weight_decay": 0.1,
        "optim_betas": (0.9, 0.95),
    }
)

# Set up log file for logging messages
log_file = os.path.join(data_dir, "olmo_training.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Create a custom callback for running inference
class InferenceCallback(Callback):
    def __init__(self, model, tokenizer_config, inference_interval, inference_prompt, log_file=None):
        self.model = model
        self.tokenizer_config = tokenizer_config
        self.inference_interval = inference_interval
        self.inference_prompt = inference_prompt
        self.inference_tokenizer = None
        self.log_file = log_file
        
    def log_message(self, message):
        """Log a message to both console and file."""
        print(message, flush=True)
        # Only write to file if the log file path is defined
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"{message}\n")
        
    def pre_train(self):
        """OLMo's pre_train method which is called before training loop starts."""
        # Load tokenizer
        from transformers import AutoTokenizer
        self.inference_tokenizer = AutoTokenizer.from_pretrained(
            "allenai/gpt-neox-olmo-dolma-v1_5",
            cache_dir=os.path.join(os.environ["TRANSFORMERS_CACHE"], "tokenizers")
        )
        
        # Token ID 0 handling is done globally now, not here
        
        # Run initial inference
        self.log_message("Running inference with untrained model...")
        self.run_inference(0)
    
    def post_step(self):
        """OLMo's post_step method which is called after each training step."""
        if self.trainer.global_step % self.inference_interval == 0 and self.trainer.global_step > 0:
            self.log_message(f"Triggering inference at step {self.trainer.global_step}")
            self.run_inference(self.trainer.global_step)
    def post_train(self):
        """OLMo's post_train method which is called at the end of training."""
        # Run final inference
        self.log_message("Running final inference with trained model...")
        self.run_inference(self.trainer.global_step)
        
    def run_inference(self, step):
    """Generate text with the model for monitoring training progress."""
        self.log_message(f"Running inference at step {step}...")
    
    # Save original model state
        was_training = self.model.training
        self.model.eval()  # Set to evaluation mode
    
    # Tokenize the prompt
        tokens = self.inference_tokenizer.encode(self.inference_prompt)
    
    # Ensure no token ID 0 in prompt
    tokens = [t for t in tokens if t != 0]
    
        token_tensor = torch.tensor([tokens], device=self.model.device)
    
    with torch.no_grad():
        # Initial forward pass
            logits = self.model(token_tensor)  # OLMo returns logits directly
        
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
                if next_token == self.tokenizer_config.eos_token_id:
                break
            
            # Add token and continue
            generated_tokens.append(next_token)
                next_input = torch.tensor([generated_tokens], device=self.model.device)
                logits = self.model(next_input)
    
    # Decode the generated text
        generated_text = self.inference_tokenizer.decode(generated_tokens)
    
    # Log output
    message = f"INFERENCE at Step {step}: \"{generated_text}\""
        self.log_message(message)
    
    # Log to W&B
    if wandb.run is not None:
        wandb.log({
            "inference/generated_text": wandb.Html(f"<p>{generated_text}</p>"),
            "inference/generated_length": len(generated_tokens) - len(tokens)
        }, step=step)
    
    # Restore original model state
    if was_training:
            self.model.train()
    
    return generated_text

# Add this instead of the monkey-patching code
inference_callback = InferenceCallback(
    model=model, 
    tokenizer_config=tokenizer_config,
    inference_interval=args.inference_interval,
    inference_prompt=args.inference_prompt,
    log_file=log_file
)

# Configure trainer
        trainer_config = TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
    cancel_check_interval=5,
            max_duration=Duration.steps(TOTAL_STEPS),
    device=str(device)
        ).with_callback(
            "wandb",
            WandBCallback(
                name=wandb.run.name if wandb.run else "olmo-190m-wiki",
                entity=None,  # Use default entity
                project="olmo-190m-wikipedia",
                enabled=True,
                cancel_check_interval=10,
    )
).with_callback(
    "inference",
    inference_callback
        )
        
        # Build trainer
        trainer = trainer_config.build(
    train_module=train_module,
    data_loader=data_loader
)
print("Trainer built successfully")

# Run training
print(f"Starting training for {TOTAL_STEPS} steps...")
        trainer.fit()
print("Training complete!")
        
        # Close wandb
        wandb.finish()
