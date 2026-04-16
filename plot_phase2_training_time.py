import json
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

with open('all_experiments_analysis.json', 'r') as f:
    data = json.load(f)

target_labels = ['B_adamw_cos', 'C_freeze50_disc', 'D_epochs40', 'E_freeze90']
exps = []
for label in target_labels:
    for exp in data:
        if exp.get('exp_label') == label and exp['model_name'] == 'resnext50_32x4d':
            exps.append(exp)
            break

labels = [e['exp_label'] for e in exps]
times_hours = [e['training_time'] / 3600.0 for e in exps]
colors = ['#e67e22', '#9b59b6', '#2ecc71', '#e74c3c']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, times_hours, color=colors)
ax.set_ylabel('Training Time (hours)')
ax.set_title('Phase 2: Training Time (NVIDIA RTX 3080)')
ax.grid(axis='y', alpha=0.3)
for bar, t in zip(bars, times_hours):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{t:.2f}h', ha='center')
ax.text(0.5, -0.15, 'Note: Baseline (A) excluded due to unstable GPU.', transform=ax.transAxes, ha='center', style='italic')
plt.tight_layout()
plt.savefig('phase2_training_time.png', dpi=150)
plt.close()