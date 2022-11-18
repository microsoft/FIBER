from .transform import (
    albef_transform,
    albef_transform_randaug,
)

_transforms = {
    "albef": albef_transform,
    "albef_randaug": albef_transform_randaug,
}


def keys_to_transforms(keys: list, size=384):
    return [_transforms[key](size=size) for key in keys]
