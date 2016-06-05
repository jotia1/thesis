#! /bin/bash

#SBATCH --job-name=demo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=joshua.arnold1@uq.net.au

JOBNAME="demo"

module load tensorflow
ARGS="--error=$JOBNAME.err --output=$JOBNAME.out"
srun --gres=gpu:1 -n1 $ARGS python demoNet.py > $JOBNAME.log

wait
