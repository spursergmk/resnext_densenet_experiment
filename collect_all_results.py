import os
import json
import pandas as pd

MANUAL_BASELINE = {
    'exp_label': 'ResNeXt_Finetune_Baseline',
    'model_name': 'resnext50_32x4d',
    'mode': 'finetune',
    'best_acc': 83.90,
    'epochs': 25,
    'freeze_ratio': 0.7,
    'optimizer': 'sgd',
    'lr_scheduler': 'step',
    'disc_lr': False,
}

def collect_all(runs_dir='runs'):
    results = [MANUAL_BASELINE]
    if os.path.exists(runs_dir):
        for exp in os.listdir(runs_dir):
            path = os.path.join(runs_dir, exp, 'final_result.json')
            if os.path.isfile(path):
                with open(path) as f:
                    data = json.load(f)
                data['exp_label'] = data.get('exp_label', exp)
                results.append(data)
    df = pd.DataFrame(results)
    cols = ['exp_label', 'model_name', 'mode', 'best_acc', 'epochs', 'freeze_ratio', 'optimizer', 'lr_scheduler']
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values('best_acc', ascending=False)
    print(df.to_string(index=False))
    df.to_csv('all_experiments_summary.csv', index=False)

if __name__ == "__main__":
    collect_all()