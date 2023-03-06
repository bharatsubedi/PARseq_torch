#!/bin/bash

num_nodes=1
gpus='7,4'
gpus_per_node=2
config="./configs/config.yaml"

CUDA_VISIBLE_DEVICES=$gpus torchrun --standalone --nnodes=$num_nodes --nproc_per_node=$gpus_per_node train.py $config