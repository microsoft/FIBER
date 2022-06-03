from .transform import (
    square_transform,
    square_transform_randaug,
)

_transforms = {
    "square": square_transform,
    "square_randaug": square_transform_randaug,
}

def keys_to_transforms(keys: list, size=384):
    return [_transforms[key](size=size) for key in keys]
