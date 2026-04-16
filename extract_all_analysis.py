import os
import json
from tensorboard.backend.event_processing import event_accumulator

RUNS_DIR = "./runs"

def load_final_results():
    results = []
    for exp in os.listdir(RUNS_DIR):
        json_path = os.path.join(RUNS_DIR, exp, "final_result.json")
        if os.path.isfile(json_path):
            with open(json_path) as f:
                data = json.load(f)
            data['log_dir'] = os.path.join(RUNS_DIR, exp)
            data['exp_folder'] = exp
            results.append(data)
    return results

def infer_label(exp_folder, data):
    name = exp_folder.lower()
    if 'freeze90' in name: return 'E_freeze90'
    if 'freeze50' in name or 'disc_lr' in name: return 'C_freeze50_disc'
    if 'adamw' in name or 'cosine' in name: return 'B_adamw_cos'
    if 'epochs40' in name: return 'D_epochs40'
    if 'scratch' in name: return f"{data.get('model_name','')}_scratch"
    if 'finetune' in name:
        # A finetune experiment with no special tag is considered baseline
        if any(k in name for k in ['freeze','adamw','cosine','epochs40','disc']):
            return 'other_finetune'
        return 'A_baseline'
    return exp_folder

def extract_tb_scalars(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    scalars = {}
    for tag in ['Train/Loss', 'Val/Loss', 'Val/Acc']:
        if tag in ea.Tags()['scalars']:
            scalars[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
    return scalars

def compute_overfitting_gap(train_losses, val_losses, last_n=5):
    if not train_losses or not val_losses: return None
    n = min(last_n, len(train_losses), len(val_losses))
    avg_train = sum(l for _,l in train_losses[-n:])/n
    avg_val = sum(l for _,l in val_losses[-n:])/n
    return avg_val - avg_train

def main():
    results = load_final_results()
    all_analysis = []
    convergence_records = []

    for r in results:
        tb = extract_tb_scalars(r['log_dir'])
        label = infer_label(r['exp_folder'], r)
        item = {
            'exp_label': label,
            'model_name': r.get('model_name'),
            'mode': r.get('mode'),
            'best_acc': r.get('best_acc'),
            'epochs': r.get('epochs'),
            'params_M': r.get('params_M'),
            'freeze_ratio': r.get('freeze_ratio'),
            'optimizer': r.get('optimizer'),
            'lr_scheduler': r.get('lr_scheduler'),
            'disc_lr': r.get('disc_lr'),
            'training_time': r.get('training_time'),
            'overfitting_gap': compute_overfitting_gap(tb.get('Train/Loss',[]), tb.get('Val/Loss',[])),
            'val_acc_curve': tb.get('Val/Acc', [])
        }
        if item['params_M']:
            item['efficiency'] = round(item['best_acc'] / item['params_M'], 2)
        all_analysis.append(item)
        for ep, acc in item['val_acc_curve']:
            convergence_records.append({'exp_label': label, 'epoch': ep, 'val_acc': acc})

    with open('all_experiments_analysis.json', 'w') as f:
        json.dump(all_analysis, f, indent=2)
    import csv
    if convergence_records:
        with open('all_convergence_data.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, ['exp_label','epoch','val_acc'])
            w.writeheader()
            w.writerows(convergence_records)
    print("Analysis data saved.")

if __name__ == "__main__":
    main()