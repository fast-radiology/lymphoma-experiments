#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=40000
#SBATCH --partition=altair-gpu
#SBATCH --time=7-0

module load cuda/10.0.130_410.48
module load python/3.7.3

export TMPDIR="/tmp/lustre_shared/${USER}/${SLURM_JOBID}"
source /home/users/${USER}/nnunet_venv/bin/activate

export nnUNet_raw_data_base="/home/users/nozdi/lymphoma_data/nnunet/"
export nnUNet_preprocessed="/home/users/nozdi/lymphoma_data/nnunet/processed/"
export RESULTS_FOLDER="/home/users/nozdi/lymphoma_data/nnunet/models/"

nnUNet_plan_and_preprocess -t 200
