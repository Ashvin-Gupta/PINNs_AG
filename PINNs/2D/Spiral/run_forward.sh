#!/bin/bash
#PBS -l select=1:ncpus=8:mem=240gb:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=50:00:00
#PBS -m ae
#PBS -N 2D_spiral
#PBS -o ./
#PBS -e ./


#Modules!
module load anaconda3/personal cuda/11.4.2
source activate PINNs_gpu

nvidia-smi
echo $CUDA_VISIBLE_DEVICES


cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
python 2D_spiral.py  

