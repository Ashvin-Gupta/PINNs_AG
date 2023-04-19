#!/bin/bash
#PBS -l select=1:ncpus=4:mem=16gb:ngpus=1
#PBS -l walltime=05:00:00
#PBS -m ae
#PBS -N 2D_model
#PBS -o ./
#PBS -e ./


#Modules!
module load anaconda3/personal cuda/11.4.2
source activate PINNs_gpu

nvidia-smi
echo $CUDA_VISIBLE_DEVICES


cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
python main.py -d 2 -f "MartaDataTrunc.mat" -m "/data/Spiral" -p

