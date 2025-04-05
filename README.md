# OLMo 190M Training ON SNELLIUS

This repository contains a script for training a lightweight 190M parameter OLMo model on Wikipedia data (3.7B tokens). The script, `190M_train.py`, provides a complete training pipeline with command-line configuration options.

## 190M_train.py

This script performs a complete training pipeline for an OLMo 190M model:

1.  Downloads Wikipedia data and tokenizes it using the OLMo tokenizer (See "Fixes Needed")
2.  Trains the OLMo 190M model with configurable parameters
3.  Logs metrics and generates sample text during training

### Key Features

-   **Command-line Configuration**: Easy parameter adjustment via command-line arguments
-   **Efficient Training**: Uses Flash Attention for faster training
-   **Data Storage Management**: Automatically manages cache directories and data storage
-   **WandB Integration**: Tracks metrics and generated text in Weights & Biases
-   **Progress Monitoring**: Includes both console logging and real-time metrics
-   **Inference Testing**: Periodically generates text from a prompt to evaluate model quality

## Setting up the Environment

To set up your environment for `190M_train.py`:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/baroian/OLMo-custom
    cd OLMo-custom
    ```

2.  **Load necessary modules (example for Snellius):**
    ```bash
    module load 2024
    module load Miniconda3/24.7.1-0
    source $EBROOTMINICONDA3/etc/profile.d/conda.sh
    # Load PyTorch separately for GPU support
    module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
    ```

3.  **Create and activate the conda environment:**
    ```bash
    conda create -n olmo python=3.9 # Specify Python version if needed
    conda activate olmo
    ```

4.  **Install base dependencies:**
    ```bash
    pip install torch wandb datasets tqdm numpy transformers
    # Ensure torch version matches the loaded module if necessary
    ```

5.  **Install OLMo Core:**
    ```bash
    git clone https://github.com/allenai/OLMo-core.git
    cd OLMo-core
    pip install -e .[all]
    cd .. # Return to OLMo-custom directory
    ```

## Preparing the Data

*   **(Work in Progress)** Data should be downloaded and preprocessed separately before starting the training job. A script like `download_data.py` will handle this. For now, all the data is downloaded when running the script, which is a waste of gpu time .

## Running the Training Job (Example on Snellius with Slurm)

1.  **Allocate GPU resources:**
    Request an interactive session on a GPU node using `salloc`. For example, to request 1 H100 GPU for 30 minutes:
    ```bash
    salloc --partition=gpu_h100 --gres=gpu:h100:1 --time=00:30:00
    # Or for an A100:
    # salloc --partition=gpu_a100 --gres=gpu:a100:1 --time=01:00:00
    ```
    You will see output similar to this, indicating the job is waiting and then granted:
    ```text
    salloc: Single-node jobs run on a shared node by default. Add --exclusive if you want to use a node exclusively.
    salloc: A full node consists of 64 CPU cores, 737280 MiB of memory and 4 GPUs and can be shared by up to 4 jobs.
    salloc: By default shared jobs get 11520 MiB of memory per CPU core, unless explicitly overridden with --mem-per-cpu, --mem-per-gpu or --mem.
    salloc: You will be charged for 1 GPUs, based on the number of CPUs, GPUs and the amount memory that you've requested.
    salloc: Pending job allocation 10991487
    salloc: job 10991487 queued and waiting for resources
    salloc: job 10991487 has been allocated resources
    salloc: Granted job allocation 10991487
    salloc: Waiting for resource configuration
    salloc: Nodes gcn135 are ready for job
    ```
    Once the allocation is granted, you need to ssh into the node, for example:
    
    ```bash
    ssh gcn135
    ```


2.  **Run the training script on the allocated node:**
    Now that you are on the compute node (e.g., `gcn135`):
    ```bash
    # Load necessary modules *on the compute node*
    module load 2024
    module load Miniconda3/24.7.1-0
    source $EBROOTMINICONDA3/etc/profile.d/conda.sh
    # Ensure the correct PyTorch module for the allocated GPU is loaded
    # Example for A100/H100:
    module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

    # Navigate to your project directory (if not already there)
    # cd /path/to/OLMo-custom/ # Adjust path if necessary

    # Activate the conda environment
    conda activate olmo

    # Run the training script
    python 190M_train.py --batch-size=24 --steps=100
    # Note: Batch size 24 is recommended (empirically best, 32 gives CUDA out of memory) for a 40GB A100 GPU. Adjust accordingly for H100 or other GPUs.
    ```

## Command-line Arguments

The script supports the following command-line arguments:

```bash
python 190M_train.py [options]

Options:
  --gpu INT          GPU ID to use (default: 0)
  --steps INT        Total training steps (default: 100)
  --batch-size INT   Batch size (default: 1)
  --prompt STR       Prompt for inference (default: "Dutch is ")
  --data-dir STR     Data directory to use (default: auto-detected)
```

### Examples

Train on GPU 0 with recommended batch size for A100 (40GB):
```bash
python 190M_train.py --batch-size 24 --steps 500
```

Train on GPU 1 with different settings:
```bash
python 190M_train.py --gpu 1 --batch-size 16 --steps 1000
```

Use a custom inference prompt and data directory:
```bash
python 190M_train.py --prompt "The meaning of life is " --data-dir /path/to/data
```

### Output

-   Model checkpoints saved in `<data_dir>/olmo_wiki_training_output_<timestamp>` (Note: Checkpointing needs fixing)
-   Training logs in `<data_dir>/olmo_training.log`
-   Metrics tracked in Weights & Biases

## Requirements

The script requires the following dependencies:

-   Python 3.9+
-   PyTorch 2.1.2+ (matching the loaded module/CUDA version)
-   OLMo Core library (`ai2-olmo`, installed from source)
-   Flash Attention (included via `OLMo-core[all]`)
-   Hugging Face libraries (`datasets`, `transformers`)
-   WandB for experiment tracking
-   Various utility libraries (`tqdm`, `numpy`, etc.)

Some dependencies are specified in `environment.yml`, but manual installation as described above is currently recommended.

## Development Notes & Future Work

### Easy Adds:
-   Flash Attention (already integrated via OLMo Core)

### Difficult Adds:
-   Layer Scaling
-   Grouped Query Attention (GQA)
-   Multihead Latent Attention

### Fixes Needed:
-   **Checkpointing**: Current checkpoint saving/loading mechanism is not fully functional.
-   **Data Pipeline**: Data download/preprocessing should be separated into a dedicated script (e.g., `download_data.py`). Training script should assume data is already processed and available.
- **Data Pipeline**: still some bugs when running, might stop working in the middle of training

### To Investigate:
-   **Inference Quality**: Inference results seem weak even after significant training (e.g., 500M tokens). Investigate potential causes (hyperparameters, implementation issues, etc.).
-   **Efficiency**: Explore low-hanging fruit for performance improvements (e.g., data loading optimization, gradient accumulation strategies).


## notes / observations

- **Througoutput**:  gtx 3090 24gb - 20k TPS (tokens per second);  a100 40gb - 75k TPS (1h = 128 SBUs); H100 95gb - 135k TPS (1h = 192 SBUs)
- H100 - 1.5 times more expensive, 1.8 times more eficient - worth it 


## Acknowledgements

This project builds upon the [OLMo Core](https://github.com/allenai/OLMo-core) library developed by the Allen Institute for Artificial Intelligence.
