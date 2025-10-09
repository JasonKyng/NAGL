#!/bin/bash

mode=$1  # "train" or "test"
gpu=$2
save_dir=$3
fold=$4
test_dataset=$5

if [ "$mode" == "train" ]; then
    sh scripts/train.sh $gpu $save_dir $fold
fi

sh scripts/test.sh $gpu $save_dir $fold $test_dataset