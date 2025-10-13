#!/bin/bash

fold=$3

if [ "$fold" == "0" ]; then
    CUDA_VISIBLE_DEVICES=$1 python test.py \
        --save_path $2 \
        --image_size 448 \
        --dataset MVTec \
        --n_shots 1 2 4 \
        --a_shots 1 \
        --num_learnable_proxies 25 \
        --num_seeds 3 \
        --eval_segm \
        --tag default \
        --data_root /path/to/dataset/mvtec
else
    if [ "$4" == "visa" ]; then
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
            --data_root /path/to/dataset/visa
    elif [ "$4" == "btad" ]; then
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
            --data_root /path/to/dataset/btad
    elif [ "$4" == "brats" ]; then
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
            --data_root /path/to/dataset/brats
    fi
fi
