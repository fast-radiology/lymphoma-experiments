#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=60000
#SBATCH --partition=altair-gpu
#SBATCH --time=7-0

module load cuda/10.0.130_410.48
module load python/3.7.3

export TMPDIR="/tmp/lustre_shared/${USER}/${SLURM_JOBID}"
source /home/users/${USER}/nnunet_venv/bin/activate

export nnUNet_raw_data_base="/home/users/nozdi/lymphoma_data/nnunet/"
export nnUNet_preprocessed="/home/users/nozdi/lymphoma_data/nnunet/processed/"
export RESULTS_FOLDER="/home/users/nozdi/lymphoma_data/nnunet/models/"
# default num threads
export nnUNet_def_n_proc=4
# augmentation num threads
export nnUNet_n_proc_DA=8
export num_cached_per_thread=2


nnUNet_train 3d_cascade_fullres nnUNetTrainerV2CascadeFullRes 200 0
