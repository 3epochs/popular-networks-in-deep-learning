#!/usr/bin/env bash
for model in alexnet vgg16 googlenet resnet152
do
    echo "python train.py -net=$model --save-dir=save_$model |& tee -a log_$model"
    python train.py -net=$model --save-dir=save_$model |& tee -a log_$model
done
