#!/bin/bash
# Phase 2: Ablation on ResNeXt-50 finetune

python run_single.py --model resnext50_32x4d --mode finetune --optimizer adamw --lr_scheduler cosine --tag adamw_cosine
python run_single.py --model resnext50_32x4d --mode finetune --freeze_ratio 0.5 --disc_lr --tag freeze50_disc_lr
python run_single.py --model resnext50_32x4d --mode finetune --epochs 40 --tag epochs40
python run_single.py --model resnext50_32x4d --mode finetune --freeze_ratio 0.9 --tag freeze90