#!/bin/sh
#PBS -q regular-g
#PBS -l select=1
#PBS -W group_list=gg17
#PBS -l walltime=1:00:00
#PBS -o med_50000_pub.out
#PBS -e med_50000_pub.err

CHECKPOINT_PATH_A=/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/users/a97006/project/bert_with_pile/checkpoint/test-160000-0.149B-iters-2M-lr-1e-3-min-1e-5-wmup-50-dcy-2M-sty-linear-gbs-1024-mbs-16-gpu-32-zero-0-mp-1-pp-1-nopp/global_step150

CHECKPOINT_PATH_B=/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/users/a97006/project/bert_with_pile/checkpoint/test-10000-0.149B-iters-2M-lr-1e-3-min-1e-5-wmup-50-dcy-2M-sty-linear-gbs-1024-mbs-16-gpu-32-zero-0-mp-1-pp-1-nopp/global_step150

PYTHON_SCRIPT_PATH="/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/250805_compare2.py ${CHECKPOINT_PATH_A} ${CHECKPOINT_PATH_B}"

module purge
module load cmake
module load gcc
module load cuda/12.6
module load cudnn/9.5.1.17
module load ompi-cuda/4.1.6-12.6

source /work/gg17/a97006/.g_bashrc
pyenv install 3.12.4 # This might take time; consider preparing an env beforehand
pyenv local 3.12.4

cd ~/env/llm-pyenv-3
source ./250/bin/activate

python ${PYTHON_SCRIPT_PATH}