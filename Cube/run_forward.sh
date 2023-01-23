#!/bin/bash
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=2
#PBS -l walltime=20:00:00
#PBS -m ae
#PBS -N Cuboid
#PBS -o ./
#PBS -e ./


#Modules!
module load anaconda3/personal
source activate /rds/general/user/ch720/home/anaconda3/envs/PINNs
ulimit -s unlimited

#(Intel compilers)
# source /opt/intel/composer_xe_2011_sp1.7.256/bin/compilervars.sh intel64
# MPT MPI settings (enabling NUMA mode and logging)
# Don't need this for non-MPI but including it for possible future use
#export MPI_DSM_DISTRIBUTE=1
#export MPI_DSM_VERBOSE=1

cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
python main.py -d 3 -f "50x50x10_week2.mat" -m "/data/Run2/" -p 
conda deactivate

