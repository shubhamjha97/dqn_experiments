#!/bin/bash
#SBATCH --job-name=dqn_double_prioritized_drq
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4

singularity exec --nv --overlay $SCRATCH/drl_hw2_overlay.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/miniconda3/bin/activate
conda activate hw2_drq
# TODO: fix
# python /scratch/sj3549/dqn_experiments/train.py 'agent.params.double_q=true' 'prioritized_replay=true' 'critic.params.dueling=false'
"