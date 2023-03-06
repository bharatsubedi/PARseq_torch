#!/bin/bash

config='./configs/config_all.yaml'
ckpt='outputs/exp_logs_baseline_all/Baseline_best_parseq_ckpt.pth'


CUDA_VISIBLE_DEVICES=1 python test.py $config --checkpoint $ckpt