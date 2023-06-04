#!/bin/bash
#PBS -l select=1:ncpus=4:mem=10gb
#PBS -l walltime=00:30:00
#PBS -m ae
#PBS -N forward_model
#PBS -o ./
#PBS -e ./


#Modules!
module load anaconda3/personal
source activate /rds/general/user/ch720/home/anaconda3/envs/PINNs
ulimit -s unlimited


cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
python FK_model_run.py 

