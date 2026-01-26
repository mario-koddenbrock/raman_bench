#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --qos=high
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%j_%x_out.log
#SBATCH --error=%j_%x_err.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=koddenb@htw-berlin.de
#SBATCH --job-name=Raman_Benchmark

# Log SLURM environment variables
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Job Name: $SLURM_JOB_NAME"
echo "SLURM Nodes: $SLURM_NODELIST"
echo "SLURM GPUs: $CUDA_VISIBLE_DEVICES"
echo "SLURM CPUs per task: $SLURM_CPUS_PER_TASK"
echo "SLURM Memory per node: $SLURM_MEM_PER_NODE"
echo "SLURM Partition: $SLURM_JOB_PARTITION"

# Log the current date and hostname
echo "Job started on $(hostname) at $(date)"

# Exit immediately if a command exits with a non-zero status.
set -e

# Load necessary modules and activate conda environment
source ~/.bashrc
conda activate raman_bench

pip install -e .

if [ $? -ne 0 ]; then
    echo "ERROR: Conda environment 'microtubule_tracking' could not be activated."
    exit 1
fi

# Print the Python executable path for debugging
which python

# Run the benchmark script
python -m raman_bench.scripts.run_benchmark --config raman_bench/config

# Log completion time
echo "Job finished at $(date)"
