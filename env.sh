#!/bin/bash
# Find and add cuda & cudnn from virtual environment wheels and add to LD_LIBRARY_PATH
# TODO: Some of the below has to be inserted manually.
# VENV_DIR="venv"
VENV_DIR="/home/advanced-brianwack/Desktop/"
# VENV_DIR="/home/jh2xl/.conda/envs/python38"
CUDNN_LIB=$(find $VENV_DIR -type d -path "*/site-packages/nvidia/cudnn/lib")
export LD_LIBRARY_PATH=$CUDNN_LIB:/usr/lib64:/usr/lib:/lib:/lib64:$LD_LIBRARY_PATH
