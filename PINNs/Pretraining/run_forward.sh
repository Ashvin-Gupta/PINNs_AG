#!/bin/bash
#PBS -l select=1:ncpus=8:mem=12g:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=04:00:00
#PBS -m ae
#PBS -N 2D_model_train
#PBS -o ./
#PBS -e ./


#Modules!
module load anaconda3/personal cuda/11.4.2
source activate PINNs_gpu

nvidia-smi
echo $CUDA_VISIBLE_DEVICES


cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
python main.py -d 2 -f "input_files/D_10" -m "/Trained/D_10/" -p 

