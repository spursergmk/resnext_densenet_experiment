import json
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

with open('all_experiments_analysis.json', 'r') as f:
    data = json.load(f)

targets = {}
for exp in data:
    model = exp['model_name']
    mode = exp['mode']
    if model == 'resnext50_32x4d' and mode == 'scratch':
        targets['ResNeXt-50\nScratch'] = exp
    elif model == 'resnext50_32x4d' and mode == 'finetune' and exp.get('exp_label') == 'A_baseline':
        targets['ResNeXt-50\nFinetune'] = exp
    elif model == 'densenet121' and mode == 'scratch':
        targets['DenseNet-121\nScratch'] = exp
    elif model == 'densenet121' and mode == 'finetune' and exp.get('exp_label') == 'A_baseline':
        targets['DenseNet-121\nFinetune'] = exp

# Accuracy bar chart
fig, ax = plt.subplots(figsize=(8, 6))
labels = list(targets.keys())
accs = [targets[l]['best_acc'] for l in labels]
colors = ['#e74c3c', '#2ecc71', '#e74c3c', '#2ecc71']
bars = ax.bar(labels, accs, color=colors)
ax.set_ylabel('Best Validation Accuracy (%)')
ax.set_title('Phase 1: Scratch vs Finetune Accuracy')
ax.grid(axis='y', alpha=0.3)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{acc:.2f}%', ha='center')
plt.tight_layout()
plt.savefig('phase1_accuracy.png', dpi=150)
plt.close()

# Convergence curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, model_key in zip(axes, ['ResNeXt-50\nScratch', 'DenseNet-121\nScratch']):
    scratch = targets[model_key]
    finetune = targets[model_key.replace('Scratch', 'Finetune')]
    if scratch['val_acc_curve']:
        epochs, accs = zip(*scratch['val_acc_curve'])
        ax.plot(epochs, accs, 'r-', label='Scratch')
    if finetune['val_acc_curve']:
        epochs, accs = zip(*finetune['val_acc_curve'])
        ax.plot(epochs, accs, 'g-', label='Finetune')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title(model_key.split('\n')[0])
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.suptitle('Phase 1: Convergence Curves')
plt.tight_layout()
plt.savefig('phase1_convergence.png', dpi=150)
plt.close()

# Efficiency bar chart
fig, ax = plt.subplots(figsize=(8, 6))
effs = [targets[l]['best_acc'] / targets[l]['params_M'] for l in labels]
bars = ax.bar(labels, effs, color=colors)
ax.set_ylabel('Parameter Efficiency (% / M)')
ax.set_title('Phase 1: Parameter Efficiency')
ax.grid(axis='y', alpha=0.3)
for bar, eff in zip(bars, effs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{eff:.2f}', ha='center')
plt.tight_layout()
plt.savefig('phase1_efficiency.png', dpi=150)
plt.close()