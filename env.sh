#!/bin/bash
# Find and add cuda & cudnn to LD_LIBRARY_PATH.
# SEARCH_DIR="venv"
# SEARCH_DIR="/home/jh2xl/.conda/envs/python38"
SEARCH_DIR="/home/advanced-brianwack/Desktop/"
ACTIVATE_PATH=$(find $SEARCH_DIR -type f -path "*/bin/activate")
source $ACTIVATE_PATH
CUDNN_LIB=$(find $SEARCH_DIR -type d -path "*/site-packages/nvidia/cudnn/lib")
export LD_LIBRARY_PATH=$CUDNN_LIB:/usr/lib64:/usr/lib:/lib:/lib64:$LD_LIBRARY_PATH
