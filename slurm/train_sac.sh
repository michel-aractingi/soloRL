#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=/tmp-network/user/maractin/logs/solo/sac/slurm_%j.out
#SBATCH --error=/tmp-network/user/maractin/logs/solo/sac/slurm_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=michel.aractingi@naverlabs.com
#SBATCH -p gpu-mono
#SBATCH -t 2000
#SBATCH -n 1
#SBATCH --cpus-per-task=8

srun nvidia-smi
srun env | grep CUDA
source /home/maractin/.bashrc
source /home/maractin/miniconda3/bin/activate py38
export LD_LIBRARY_PATH=/nfs/core/cuda/9.1/lib64:$LD_LIBRARY_PATH

LOGDIR=/tmp-network/user/maractin/logs/solo/sac
cd $LOGDIR
CODEDIR=/home/maractin/Workspace/soloRL

python -u  $CODEDIR/training/train_sac.py --num-agents 64  --logdir $LOGDIR/ --log-interval 1000 --save-interval 2000 --timestamp ${SLURM_JOB_ID} --learning-rate 0.00025 --config-file $CODEDIR/configs/basic.yaml --seed 123123 $1 $2 $3 $4 $5 $6 $7 $8 $9 

