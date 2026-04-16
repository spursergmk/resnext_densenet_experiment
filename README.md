# CNN Transfer Learning Benchmark

This repository contains code for comparing training-from-scratch and fine-tuning on CIFAR-100 using ResNeXt-50 and DenseNet-121. It also includes ablation studies on fine-tuning strategies.

# How to run the experiments

## 1. Install dependencies
   pip install -r requirements.txt

## 2. Prepare dataset
   Place CIFAR-100 under ./data or set download=True in dataset.py.

## 3. Run Phase 1 (scratch vs finetune for both models)
   bash run_all_phase1.sh

   This runs:
   - ResNeXt-50 scratch (60 epochs)
   - ResNeXt-50 finetune (25 epochs)
   - DenseNet-121 scratch (60 epochs)
   - DenseNet-121 finetune (25 epochs)

## 4. Run Phase 2 (ablation on ResNeXt-50 finetune)
   bash run_ablation.sh

   This runs:
   - B: AdamW + Cosine annealing
   - C: Freeze 50% + discriminative LR
   - D: 40 epochs
   - E: Freeze 90%

## 5. Extract results and generate figures
   bash run_all_analysis.sh

   This will produce:
   - all_experiments_analysis.json
   - all_convergence_data.csv
   - PNG figures for Phase 1 and Phase 2
   - all_experiments_summary.csv

## Custom experiments
   python run_single.py --model resnext50_32x4d --mode finetune --freeze_ratio 0.8 --epochs 35 --tag my_run

## Notes
   Training times are not strictly comparable due to mixed GPU models (3080/4080) on the shared platform.