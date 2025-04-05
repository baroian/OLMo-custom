#!/bin/bash
#SBATCH --partition=gpu_a100       # Specify the partition (queue)
#SBATCH --gres=gpu:a100:1         # Request 1 A100 GPU
#SBATCH --time=04:00:00         # Set a maximum runtime of 4 hours
#SBATCH --job-name=olmo_190m_train # Sensible job name
#SBATCH --output=olmo_train.%j.log  # Log file for stdout/stderr (%j = Job ID)
#SBATCH --cpus-per-task=4         # Request CPU cores (adjust if needed)
#SBATCH --mem=40G                 # Request memory (adjust if needed)

echo "----------------------------------------------------"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node(s): $SLURM_JOB_NODELIST"
echo "Run Directory: $(pwd)"
echo "Job started on $(hostname) at $(date)"
echo "----------------------------------------------------"

# Load necessary modules
echo "Loading modules..."
module load 2024
module load Miniconda3/24.7.1-0
echo "Modules loaded."

# Initialize Conda for bash shell
echo "Initializing Conda..."
source $EBROOTMINICONDA3/etc/profile.d/conda.sh
echo "Conda initialized."


# Activate the Conda environment
echo "Activating conda environment: olmo..."
conda activate olmo
if [ $? -ne 0 ]; then
  echo "Error activating conda environment 'olmo'. Exiting."
  exit 1
fi
echo "Conda environment activated."


# Run the Python training script
# NOTE: Corrected 'python python' to 'python' from your example.
echo "Starting Python training script (190M_train.py)..."
python 190M_train.py --batch-size=24 --steps=40000

EXIT_CODE=$?
echo "----------------------------------------------------"
echo "Python script finished with exit code $EXIT_CODE"
echo "Job finished at $(date)"
echo "----------------------------------------------------"

exit $EXIT_CODE