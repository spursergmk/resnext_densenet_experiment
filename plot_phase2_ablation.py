import json
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

with open('all_experiments_analysis.json', 'r') as f:
    data = json.load(f)

target_labels = ['A_baseline', 'B_adamw_cos', 'C_freeze50_disc', 'D_epochs40', 'E_freeze90']
exps = []
for label in target_labels:
    for exp in data:
        if exp.get('exp_label') == label and exp['model_name'] == 'resnext50_32x4d':
            exps.append(exp)
            break

labels = [e['exp_label'] for e in exps]
accs = [e['best_acc'] for e in exps]
colors = ['#3498db', '#e67e22', '#9b59b6', '#2ecc71', '#e74c3c']

# Accuracy bars
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, accs, color=colors)
ax.set_ylabel('Best Validation Accuracy (%)')
ax.set_title('Phase 2: ResNeXt-50 Finetune Ablation')
ax.grid(axis='y', alpha=0.3)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{acc:.2f}%', ha='center')
plt.tight_layout()
plt.savefig('phase2_ablation_accuracy.png', dpi=150)
plt.close()

# Convergence curves
fig, ax = plt.subplots(figsize=(12, 7))
for exp, color in zip(exps, colors):
    curve = exp.get('val_acc_curve', [])
    if curve:
        epochs, vals = zip(*curve)
        ax.plot(epochs, vals, label=exp['exp_label'], color=color, linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Accuracy (%)')
ax.set_title('Phase 2: Ablation Convergence Curves')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('phase2_ablation_convergence.png', dpi=150)
plt.close()