#!/bin/bash
#SBATCH --job-name=dqn
#SBATCH --open-mode=append
#SBATCH --output=./%x_%j.out
#SBATCH --error=./%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8

singularity exec --nv --overlay $SCRATCH/drl_hw2_overlay.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/miniconda3/bin/activate
conda activate hw2_drq
python -u /scratch/sj3549/dqn_experiments/train.py 'agent.params.double_q=false' 'prioritized_replay=false' 'critic.params.dueling=false' 'env=Breakout' &
python -u /scratch/sj3549/dqn_experiments/train.py 'agent.params.double_q=false' 'prioritized_replay=false' 'critic.params.dueling=false' 'env=Pong' &
python -u /scratch/sj3549/dqn_experiments/train.py 'agent.params.double_q=false' 'prioritized_replay=false' 'critic.params.dueling=false' 'env=SpaceInvaders' &
wait
"