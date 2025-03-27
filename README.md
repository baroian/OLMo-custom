# OLMo 190M Training

<<<<<<< HEAD
This repository contains a script for training a lightweight 190M parameter OLMo model on Wikipedia data. The script, `190M_train.py`, provides a complete training pipeline with command-line configuration options.
=======
This repository contains scripts for training 190M parameters OLMo language models on Wikipedia dataset (3.7B tokens). The primary script, `OLMo2-190M-w.py`, trains a lightweight 190M parameter OLMo model on a small subset of Wikipedia data.

## OLMo2-190M-wiki-subset

This script performs a complete training pipeline for a small OLMo model:

1. Downloads a tiny subset of Wikipedia data (0.2% or less, depending on scale factor)
2. Tokenizes it using the GPT NeoX OLMo Dolma v1.5 tokenizer
3. Converts it to the `.npy` format expected by OLMo
4. Trains a 190M parameter model on this data
5. Periodically runs inference to showcase model progress

### Key Features

- **Customizable Scale**: Adjust `scale_factor` to control training duration (1 = 1000 steps)
- **Batch Size Control**: Modify `batch_size_factor` to adjust effective batch size
- **Data Storage Management**: All cache and output directories are configured to use `/data1` to avoid quota issues
- **WandB Integration**: Tracks metrics and generated text in Weights & Biases
- **Progress Monitoring**: Includes both console logging and progress bar during training
- **Inference Testing**: Periodically generates text from a prompt to evaluate model quality
>>>>>>> 72276e0064b6e9707cbac2f9827bd5f2c38abdcb

## 190M_train.py

This script performs a complete training pipeline for an OLMo 190M model:

1. Downloads Wikipedia data and tokenizes it using the OLMo tokenizer
2. Trains the OLMo 190M model with configurable parameters
3. Logs metrics and generates sample text during training
4. Automatically adapts to different computing environments

### Key Features

- **Command-line Configuration**: Easy parameter adjustment via command-line arguments
- **Efficient Training**: Uses Flash Attention for faster training
- **Data Storage Management**: Automatically manages cache directories and data storage
- **WandB Integration**: Tracks metrics and generated text in Weights & Biases
- **Progress Monitoring**: Includes both console logging and real-time metrics
- **Inference Testing**: Periodically generates text from a prompt to evaluate model quality

### Command-line Arguments

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

Train on GPU 0 with default settings:
```bash
python 190M_train.py
```

Train on GPU 1 with larger batch size and more steps:
```bash
python 190M_train.py --gpu 1 --batch-size 4 --steps 500
```

Use a custom inference prompt and data directory:
```bash
python 190M_train.py --prompt "The meaning of life is " --data-dir /path/to/data
```

### Output

- Model checkpoints saved in `<data_dir>/olmo_wiki_training_output_<timestamp>`
- Training logs in `<data_dir>/olmo_training.log`
- Metrics tracked in Weights & Biases

## Setting up the Environment

To set up your environment for 190M_train.py:

1. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate olmo
   ```

2. Verify the environment is set up correctly:
   ```bash
   python3 check_env.py
   ```

3. If you need to install OLMo-core from source (instead of via pip):
   ```bash
   git clone https://github.com/allenai/OLMo-core.git
   cd OLMo-core
   pip install -e .
   cd ..
   ```

The `check_env.py` script will verify that all required dependencies are installed and that your GPU is accessible.

## Requirements

The script requires the following dependencies:

- Python 3.9+
- PyTorch 2.5.1+
- OLMo Core library (ai2-olmo)
- Flash Attention for efficient transformer operations
- Hugging Face libraries (datasets, transformers)
- WandB for experiment tracking
- Various utility libraries (tqdm, numpy, etc.)

All dependencies are specified in the `environment.yml` file.

## Acknowledgements

This project builds upon the [OLMo Core](https://github.com/allenai/OLMo-core) library developed by the Allen Institute for Artificial Intelligence. 
