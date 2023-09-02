#!/bin/bash
# Request 64 gigabytes of real memory (RAM) 8 cores *8G = 64
#SBATCH --mem=64G
# Request 8 cores
#SBATCH --cpus-per-task=8
# Email notifications
#SBATCH --mail-user=jbao12@sheffield.ac.uk
# Email notifications
#SBATCH --mail-type=ALL
# Change the name of the output log file.
#SBATCH --output=./outputs/bessemer_asr_train_asr_%j.txt
# Rename the job's name
#SBATCH --comment=bessemer_asr_train_asr

# GPUs
#SBATCH --account=clarity
#SBATCH --partition=clarity
#SBATCH --gres=gpu:1
#SBATCH --time=6-23:59:59

# Load the modules required by our program
# module load libsndfile-1.0.28-gcc-4.8.5-worvtuj
module load Anaconda3/2019.07
module load CUDA/10.0.130
# module load imkl/2019.1.144-iimpi-2019a
source activate clarity

# Set the OPENMP_NUM_THREADS environment variable to 4
# This is needed to ensure efficient core usage.
export OMP_NUM_THREADS=$NSLOTS

# Run the program
# Train ASR model
python train_asr.py transformer_cpc2.yaml