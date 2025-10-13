#!/bin/bash

if [ "$3" == "mvtec" ]; then
    CUDA_VISIBLE_DEVICES=$1 python test.py \
        --save_path $2 \
        --image_size 224 \
        --dataset MVTec \
        --n_shots 1 2 4 \
        --a_shots 1 \
        --num_learnable_proxies 25 \
        --num_seeds 3 \
        --eval_segm \
        --tag default \
        --data_root /gpfs/work/int/yuexinwang23/Datasets/mvtec
elif [ "$3" == "visa" ]; then
    CUDA_VISIBLE_DEVICES=$1 python test.py \
        --save_path $2 \
        --image_size 448 \
        --dataset VisA \
        --n_shots 1 2 4 \
        --a_shots 1 \
        --num_learnable_proxies 25 \
        --num_seeds 3 \
        --eval_segm \
        --tag default \
        --data_root /gpfs/work/int/yuexinwang23/Datasets/visa
elif [ "$3" == "btad" ]; then
    CUDA_VISIBLE_DEVICES=$1 python test.py \
        --save_path $2 \
        --image_size 448 \
        --dataset BTAD \
        --n_shots 1 2 4 \
        --a_shots 1 \
        --num_learnable_proxies 25 \
        --num_seeds 3 \
        --eval_segm \
        --tag default \
        --data_root /gpfs/work/int/yuexinwang23/Datasets/btad
elif [ "$3" == "brats" ]; then
    CUDA_VISIBLE_DEVICES=$1 python test.py \
        --save_path $2 \
        --image_size 448 \
        --dataset BraTS \
        --n_shots 1 2 4 \
        --a_shots 1 \
        --num_learnable_proxies 25 \
        --num_seeds 3 \
        --eval_segm \
        --tag default \
        --data_root /gpfs/work/int/yuexinwang23/Datasets/brats
fi
