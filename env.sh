#!/bin/bash
# Find and add cuda & cudnn from virtual environment wheels and add to LD_LIBRARY_PATH
# TODO: Some of the below has to be inserted manually.
# VENV_DIR="venv"
VENV_DIR="/home/jh/.conda/envs/temp/lib/python3.9"
CUDNN_LIB=$(find $VENV_DIR -type d -path "*/site-packages/nvidia/cudnn/lib")
# CUDA_LIB=/usr/lib/nvidia/lib
CUDA_LIB="/lib/wsl/lib"
export LD_LIBRARY_PATH=$CUDNN_LIB:$CUDA_LIB:$LD_LIBRARY_PATH
