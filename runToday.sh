#!/bin/sh
mkdir -p ../model/lstmfnn/epoch10
mkdir -p ../model/lstmfnn/epoch50
mkdir -p ../model/lstmfnn/epoch100

CUDA_VISIBLE_DEVICES=0 nohup python lstmfnn.py > lstmfnn1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python lstmfnn.py > lstmfnn2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python lstmfnn.py > lstmfnn3.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python lstmfnn.py > lstmfnn4.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python lstmfnn.py > lstmfnn5.log 2>&1 &
