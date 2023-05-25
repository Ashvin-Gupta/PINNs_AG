#!/bin/bash
#PBS -l select=1:ncpus=8:mem=15gb:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=00:30:00
#PBS -m ae
#PBS -N forward_model
#PBS -o ./
#PBS -e ./


#Modules!
module load anaconda3/personal cuda/11.4.2
#source activate PINNs_gpu
#source activate /rds/general/user/ch720/home/anaconda3/envs/PINNs
#module purge
#module load tools/prod
#module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1 
#module add scikit-learn/1.0.2-foss-2021b
source activate PINNs_gpu
#module list

nvidia-smi
echo $CUDA_VISIBLE_DEVICES


cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
python mainload.py -d 2 -f "input_files/D_08.mat" -m "/Trained/D_08/" -p 

