#!/bin/bash
#SBATCH --job-name=dqn_double_prioritized_drq
#SBATCH --open-mode=append
#SBATCH --output=./%x_%j.out
#SBATCH --error=./dqn_double_prioritized_drq_%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8

singularity exec --nv --overlay $SCRATCH/drl_hw2_overlay.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/miniconda3/bin/activate
conda activate hw2_drq

# all
python -u /scratch/sj3549/dqn_experiments/train.py 'agent.params.double_q=true' 'prioritized_replay=false' 'critic.params.dueling=true' 'agent.params.drq_k=2' 'agent.params.drq_m=2' 'aug_type=all' 'env=Breakout' &
python -u /scratch/sj3549/dqn_experiments/train.py 'agent.params.double_q=true' 'prioritized_replay=false' 'critic.params.dueling=true' 'agent.params.drq_k=2' 'agent.params.drq_m=2' 'aug_type=all' 'env=Pong' &
python -u /scratch/sj3549/dqn_experiments/train.py 'agent.params.double_q=true' 'prioritized_replay=false' 'critic.params.dueling=true' 'agent.params.drq_k=2' 'agent.params.drq_m=2' 'aug_type=all' 'env=SpaceInvaders' &

wait
"
