#! /bin/bash

#SBATCH --job-name=pilotA
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=joshua.arnold1@uq.net.au

JOBNAME="pilotA"

module load tensorflow
ARGS="--error=$JOBNAME.err --output=$JOBNAME.out"
srun --gres=gpu:1 -n1 $ARGS python batchNetRun.py > $JOBNAME.log

wait
