import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, SequentialLR, LinearLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import time
import json
import shutil
from config import Config as cfg
from models import build_model, freeze_backbone

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('best_acc', 0.0)

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({
            'Loss': f"{running_loss/(batch_idx+1):.3f}",
            'Acc': f"{100.*correct/total:.2f}%"
        })

        if writer:
            step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), step)
            writer.add_scalar('Train/Acc', 100.*correct/total, step)

    return running_loss/len(dataloader), 100.*correct/total

def validate(model, dataloader, criterion, device, epoch, writer=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'Loss': f"{running_loss/(batch_idx+1):.3f}",
                'Acc': f"{100.*correct/total:.2f}%"
            })

    val_loss = running_loss/len(dataloader)
    val_acc = 100.*correct/total

    if writer:
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Acc', val_acc, epoch)

    return val_loss, val_acc

def train_model(model_name, mode, train_loader, val_loader, device,
                epochs_override=None, freeze_ratio=0.7, optimizer_type='sgd',
                lr_scheduler_type='step', use_disc_lr=False, exp_tag=''):
    
    base_name = cfg.get_exp_name(model_name, mode)
    exp_name = f"{base_name}_{exp_tag}" if exp_tag else base_name

    checkpoint_dir = os.path.join(cfg.SAVE_DIR, exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(cfg.LOG_DIR, exp_name)
    writer = SummaryWriter(log_dir)

    shutil.copy('config.py', os.path.join(log_dir, 'config_snapshot.py'))

    print(f"\n{'='*60}")
    print(f"Training: {model_name} | Mode: {mode}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*60}")

    model = build_model(model_name, num_classes=cfg.NUM_CLASSES, mode=mode)
    if mode == "finetune":
        model = freeze_backbone(model, model_name, freeze_ratio=freeze_ratio)
    model = model.to(device)

    params_m = count_parameters(model)
    print(f"Model Parameters: {params_m:.2f} M")
    with open(os.path.join(log_dir, 'model_info.json'), 'w') as f:
        json.dump({'model_name': model_name, 'mode': mode, 'params_M': params_m}, f)

    criterion = nn.CrossEntropyLoss()
    
    epochs = epochs_override if epochs_override is not None else (
        cfg.EPOCHS_FINETUNE if mode == 'finetune' else cfg.EPOCHS_SCRATCH)

    if use_disc_lr and mode == 'finetune':
        backbone_params = []
        classifier_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'fc' in name or 'classifier' in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
        param_groups = [
            {'params': backbone_params, 'lr': cfg.LEARNING_RATE_FINETUNE * 0.1},
            {'params': classifier_params, 'lr': cfg.LEARNING_RATE_FINETUNE}
        ]
    else:
        param_groups = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(param_groups, lr=cfg.LEARNING_RATE_FINETUNE, weight_decay=cfg.WEIGHT_DECAY)
    else:
        optimizer = optim.SGD(param_groups, lr=cfg.LEARNING_RATE_FINETUNE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)

    if lr_scheduler_type == 'cosine':
        warmup_epochs = 5
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    else:
        milestones = [int(epochs * 0.6), int(epochs * 0.8)]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    start_epoch = 1
    best_acc = 0.0
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        start_epoch, best_acc = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        start_epoch += 1

    epoch_times = []
    total_start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, writer)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, Time={epoch_time:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"  -> Best model updated (Acc={best_acc:.2f}%)")

        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }, checkpoint_path)

    total_time = time.time() - total_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0

    result = {
        'model_name': model_name,
        'mode': mode,
        'best_acc': best_acc,
        'training_time': total_time,
        'epochs': epochs,
        'avg_epoch_time': avg_epoch_time,
        'params_M': params_m,
        'exp_name': exp_name,
        'freeze_ratio': freeze_ratio,
        'optimizer': optimizer_type,
        'lr_scheduler': lr_scheduler_type,
        'disc_lr': use_disc_lr,
    }
    with open(os.path.join(log_dir, 'final_result.json'), 'w') as f:
        json.dump(result, f, indent=4)

    writer.close()
    return result