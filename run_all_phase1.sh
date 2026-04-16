#!/bin/bash
# Phase 1: Scratch vs Finetune for both models

python run_single.py --model resnext50_32x4d --mode scratch
python run_single.py --model resnext50_32x4d --mode finetune --tag baseline
python run_single.py --model densenet121 --mode scratch
python run_single.py --model densenet121 --mode finetune --tag baseline