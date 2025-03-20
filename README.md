# OLMo Custom Training

This repository contains scripts for training custom OLMo language models on specific datasets. The primary script, `OLMo2-190M-wiki-subset.py`, trains a lightweight 190M parameter OLMo model on a small subset of Wikipedia data.

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

## Requirements

The script requires the following dependencies:

- Python 3.9+
- PyTorch 2.5.1+
- OLMo Core library (requires installation in development mode)
- Hugging Face libraries (datasets, transformers)
- WandB for experiment tracking
- Various utility libraries (tqdm, numpy, etc.)

See `requirements.txt` for the complete list of dependencies.

## Usage

1. Set up the environment:
   ```bash
   conda env create -f environment.yml
   conda activate olmo
   ```

2. Install OLMo Core:
   ```bash
   cd OLMo-core
   pip install -e .
   ```

3. Run the script:
   ```bash
   python OLMo2-190M-wiki-subset.py
   ```

### Configuration

The script includes several customizable parameters:

- `scale_factor`: Controls training duration and dataset size (1 = 1000 steps, ~1M tokens)
- `batch_size_factor`: Multiplier for batch size (default: 4)
- `data_dir`: Base directory for all data and cache files
- `inference_prompt`: Starting text for generation tests

## Output

The script generates several outputs:

- Trained model checkpoints (saved every 100 steps)
- WandB logs (metrics, loss curves, generated text samples)
- Console output with detailed progress information
- Log file at `{data_dir}/wiki_training.log`
- Generated text examples in `{data_dir}/inference_outputs.txt`

## Acknowledgements

This project builds upon the [OLMo Core](https://github.com/allenai/OLMo-core) library developed by the Allen Institute for Artificial Intelligence. 