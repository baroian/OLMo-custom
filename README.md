# OLMo-custom

Train the 190 M-parameter [OLMo](https://allenai.org/olmo) language model locally on a **small, self-contained dataset** (default: a Wikipedia sample) using the open-source *OLMo-core* library.

This repository is meant for quick experimentation and debugging – it keeps only the pieces required to run end-to-end training on a single machine (CPU/GPU or multi-GPU via FSDP).

## Repository layout (high level)

• `train.py` – kicks off data loading, model build and training.
• `configs/` – YAML files with training hyper-parameters. Pass one with `--config <name>`.
• `utils/` – thin wrappers around OLMo-core to build the dataloader, model, special-token handling and inference callback.
• `data_utils/` – scripts that can download, tokenise and save datasets to simple `.npy` files.
• `olmo_core/` – vendored snapshot of the upstream OLMo-core library (no changes required on your side).

## Quick start

1. Clone this repo and move the upstream library inside it (as described below) or run `git submodule add` if you prefer.
2. Create the environment and install dependencies:

```bash
chmod +x setup.sh
./setup.sh         # creates a venv and installs the pinned requirements
```

3. Prepare a tiny dataset (optional – the first run of `train.py` will do it automatically):

```bash
python -m data_utils.download_and_tokenizeV2 --config configs/config.yaml
```

4. Launch training on **one** GPU/CPU:

```bash
torchrun train.py --steps 50 --batch-size 2 --prompt "The universe is "
```

5. Launch training on **multiple** GPUs with Fully-Sharded Data Parallel:

```bash
torchrun --nproc_per_node 2 train.py --config base18L --steps 50 --batch-size 2
```

Checkpoints are written to `<data_dir>/checkpoints/` (see `config.yaml`).

## Using your own data

Change the paths and sampling proportions in `configs/config.yaml` and run one of the `data_utils/*tokenize*.py` scripts to generate the `.npy` files expected by the training script.

## Upstream library

The directory `olmo_core/` must contain the **`olmo_core`** Python package. You can copy it manually:

```bash
# from the cloned OLMo-core repo
cp -r OLMo-core/src/olmo_core <this-repo>/olmo_core
```

or keep it as a submodule.

---
This README intentionally stays brief – dive into the code for details or open an issue if something is unclear.
