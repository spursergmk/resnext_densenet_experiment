import argparse
import torch
from config import Config as cfg
from dataset import get_dataloaders
from train import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=cfg.MODELS)
    parser.add_argument('--mode', type=str, required=True, choices=cfg.TRAINING_MODES)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--freeze_ratio', type=float, default=0.7)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'])
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['step', 'cosine'])
    parser.add_argument('--disc_lr', action='store_true')
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(mode=args.mode)

    result = train_model(
        args.model, args.mode, train_loader, val_loader, device,
        epochs_override=args.epochs,
        freeze_ratio=args.freeze_ratio,
        optimizer_type=args.optimizer,
        lr_scheduler_type=args.lr_scheduler,
        use_disc_lr=args.disc_lr,
        exp_tag=args.tag
    )