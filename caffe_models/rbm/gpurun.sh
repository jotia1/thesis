#!/bin/bash
#
# gres_test.bash
# Submit as follows:
# sbatch --gres=gpu:1 -n1 -N1-1 gres_test.bash
#
srun --gres=gpu:1 -n1 --exclusive caffe train -solver /home/Student/s4290365/thesis/caffe_models/rbm/snet_solver.prototxt &
wait
