import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import Config as cfg

def get_transforms(mode="scratch"):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if mode == "scratch":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform

def get_dataloaders(mode="scratch", batch_size=None, num_workers=None):
    if batch_size is None:
        batch_size = cfg.BATCH_SIZE
    if num_workers is None:
        num_workers = cfg.NUM_WORKERS

    train_transform, val_transform = get_transforms(mode)

    train_dataset = datasets.CIFAR100(
        root=cfg.DATA_ROOT, train=True,
        download=False, transform=train_transform
    )
    val_dataset = datasets.CIFAR100(
        root=cfg.DATA_ROOT, train=False,
        download=False, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader