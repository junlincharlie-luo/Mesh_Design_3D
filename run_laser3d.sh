#!/bin/bash
#SBATCH --job-name=laser3d
#SBATCH --partition=gpu-ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=laser3d_%j.out
#SBATCH --error=laser3d_%j.err

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "========================================"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fenics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd $SLURM_SUBMIT_DIR

# Run FEniCS simulation (serial — 600K DOFs, 200 time steps)
python3 laser_single_track.py \
    --dx 0.125e-3 \
    --dt 0.125e-3 \
    --t_final 0.025 \
    --laser_sigma 2e-4 \
    --output_folder output_single_track \
    --savefreq 5

echo "========================================"
echo "End:    $(date)"
echo "========================================"
