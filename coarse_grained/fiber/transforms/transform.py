"""
    From https://github.com/salesforce/ALBEF/
"""
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip
from .randaug import RandomAugment


def albef_transform(size=384):
    return Compose(
        [
            Resize((size, size), interpolation=Image.BICUBIC),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def albef_transform_randaug(size=384):
    return Compose(
        [
            RandomResizedCrop(size, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            RandomHorizontalFlip(),
            RandomAugment(
                2,
                7,
                isPIL=True,
                augs=[
                    "Identity",
                    "AutoContrast",
                    "Equalize",
                    "Brightness",
                    "Sharpness",
                    "ShearX",
                    "ShearY",
                    "TranslateX",
                    "TranslateY",
                    "Rotate",
                ],
            ),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
