#! /bin/bash  

#SBATCH --job-name=attacc
#SBATCH --partition=gpu                                                   
#SBATCH --gres=gpu:1          
#SBATCH --time=1-00:00:00     
#SBATCH --mail-type=end       
#SBATCH --mail-user=joshua.arnold1@uq.net.au 


MATLAB_LINE="batchAedat2NetIn, exit"

matlab -nosplash -nodisplay -r "$MATLAB_LINE" > matlab_att.log
