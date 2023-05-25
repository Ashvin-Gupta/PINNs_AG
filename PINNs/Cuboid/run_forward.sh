#!/bin/bash
#PBS -l select=1:ncpus=28:mem=240gb:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=16:00:00
#PBS -m ae
#PBS -N 3D_model_train
#PBS -o ./
#PBS -e ./


#Modules!
module load anaconda3/personal cuda/11.4.2
source activate PINNs_gpu

nvidia-smi
echo $CUDA_VISIBLE_DEVICES


cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
python 3D_cube.py  

