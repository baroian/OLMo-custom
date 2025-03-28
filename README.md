# OLMo 190M Training

This repository contains a script for training a lightweight 190M parameter OLMo model on Wikipedia data (3.7B tokens). The script, `190M_train.py`, provides a complete training pipeline with command-line configuration options.

## 190M_train.py

This script performs a complete training pipeline for an OLMo 190M model:

1. Downloads Wikipedia data and tokenizes it using the OLMo tokenizer
2. Trains the OLMo 190M model with configurable parameters
3. Logs metrics and generates sample text during training

### Key Features

- **Command-line Configuration**: Easy parameter adjustment via command-line arguments
- **Efficient Training**: Uses Flash Attention for faster training
- **Data Storage Management**: Automatically manages cache directories and data storage
- **WandB Integration**: Tracks metrics and generated text in Weights & Biases
- **Progress Monitoring**: Includes both console logging and real-time metrics
- **Inference Testing**: Periodically generates text from a prompt to evaluate model quality



## Setting up the Environment

To set up your environment for 190M_train.py:

1. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate olmo
   ```
2. Train on GPU 0 with default settings:
```bash
python 190M_train.py
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
