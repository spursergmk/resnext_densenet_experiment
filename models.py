import torch.nn as nn
from torchvision import models

def build_model(model_name="resnext50_32x4d", num_classes=100, mode="scratch"):
    if model_name == "resnext50_32x4d":
        weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2 if mode == "finetune" else None
        model = models.resnext50_32x4d(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if mode == "finetune" else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model

def freeze_backbone(model, model_name, freeze_ratio=0.7):
    if model_name == "resnext50_32x4d":
        layers = list(model.children())
        backbone_layers = layers[:-1]
        num_freeze = int(len(backbone_layers) * freeze_ratio)
        for layer in backbone_layers[:num_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

    elif model_name == "densenet121":
        submodules = list(model.features.children())
        num_freeze = int(len(submodules) * freeze_ratio)
        for module in submodules[:num_freeze]:
            for param in module.parameters():
                param.requires_grad = False

    else:
        raise ValueError(f"Unsupported model for freezing: {model_name}")

    return model