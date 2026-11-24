import torch
from torchvision import transforms as T
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_model(n_classes: int = 6, pretrained: bool = True) -> torch.nn.Module:
    net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
    net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, n_classes)
    return net


def get_transforms(train: bool):
    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    if not train:
        # The whole photo is kept: a centre crop would cut vegetables out of tall photos.
        return T.Compose([T.Resize((SIZE, SIZE)), *norm])
    return T.Compose([
        # Stores shoot from different distances and crop vegetables at the frame edge.
        T.RandomResizedCrop(SIZE, scale=(0.5, 1.0), ratio=(0.6, 1.7)),
        T.RandomHorizontalFlip(),
        # Lighting differs; hue barely moves because colour separates classes.
        T.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.02),
        T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.2),
        *norm,
        # Price tags and neighbours hide parts of a vegetable.
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
