#!/bin/sh

nohup CUDA_VISIBLE_DEVICES=0 python lstmonly.py > lstmonly.log 2>&1 &
nohup CUDA_VISIBLE_DEVICES=1 python lstmfnn.py > lstmfnn.log 2>&1 &
nohup CUDA_VISIBLE_DEVICES=2 python mlp.py > mlp.log 2>&1 &
nohup CUDA_VISIBLE_DEVICES=3 python mlpfnn.py > mlpfnn.log 2>&1 &