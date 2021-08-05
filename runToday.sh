#!/bin/sh
mkdir -p model/mlp
mkdir -p model/mlpfnn
mkdir -p model/lstm
mkdir -p model/lstmfnn
CUDA_VISIBLE_DEVICES=0 nohup python lstmonly.py > lstmonly.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python lstmfnn.py > lstmfnn.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python mlp.py > mlp.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python mlpfnn.py > mlpfnn.log 2>&1 &