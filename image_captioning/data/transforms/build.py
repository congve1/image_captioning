import torchvision.transforms as transforms

from image_captioning.config import cfg

def build_transforms():
    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [
            transforms.Resize([cfg.INPUT.SIZE, cfg.INPUT.SIZE]),
            transforms.ToTensor(),
            normalize_transform
        ]
    )
    return transform