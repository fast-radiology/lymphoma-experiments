#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#SBATCH --partition=tesla
#SBATCH --time=7-0

module load cuda/10.0.130_410.48
module load python/3.7.3

export TMPDIR="/tmp/lustre_shared/${USER}/${SLURM_JOBID}"
export DATA_PATH="$HOME/lymphoma_data/prediction_data"
export RESULTS_PATH="$HOME/lymphoma_data"
export MODEL_PATH="$HOME/lymphoma_data/models/best_metric_model.pth"
source /home/users/${USER}/monai/bin/activate


cat << EOF
-------------------------------------------------------------------------------

Start of calculations [$(date)]
EOF

export OMP_NUM_THREADS=10
python $HOME/lymphoma/run_evaluation.py

cat << EOF
-------------------------------------------------------------------------------

End of calculations [$(date)].

-------------------------------------------------------------------------------
EOF
