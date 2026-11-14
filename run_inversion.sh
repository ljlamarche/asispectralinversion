#!/bin/bash -l

#SBATCH --job-name=gneiss_rgb
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=gneiss-%j.log

# Stop if a command fails
set -e

# Go to the project directory
cd "$SLURM_SUBMIT_DIR"

# Avoid extra CPU threads inside each worker
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Use the existing Conda environment
PYTHON_BIN="$HOME/.conda/envs/gneiss-rgb/bin/python"

# Check that Python exists
if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python environment not found: $PYTHON_BIN"
    exit 1
fi

# Print job information
echo "Starting GNEISS inversion"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

# Run your Python script
"$PYTHON_BIN" -u gneiss_rgb.py

echo "Finished: $(date)"
