"""Extends default ops to accept optional parameters."""
from functools import partial

from timm.data.auto_augment import _LEVEL_DENOM, _randomly_negate, LEVEL_TO_ARG, NAME_TO_OP, rotate


def rotate_expand(img, degrees, **kwargs):
    """Rotate operation with expand=True to avoid cutting off the characters"""
    kwargs['expand'] = True
    return rotate(img, degrees, **kwargs)


def _level_to_arg(level, hparams, key, default):
    magnitude = hparams.get(key, default)
    level = (level / _LEVEL_DENOM) * magnitude
    level = _randomly_negate(level)
    return level,


def apply():
    # Overrides
    NAME_TO_OP.update({
        'Rotate': rotate_expand
    })
    LEVEL_TO_ARG.update({
        'Rotate': partial(_level_to_arg, key='rotate_deg', default=30.),
        'ShearX': partial(_level_to_arg, key='shear_x_pct', default=0.3),
        'ShearY': partial(_level_to_arg, key='shear_y_pct', default=0.3),
        'TranslateXRel': partial(_level_to_arg, key='translate_x_pct', default=0.45),
        'TranslateYRel': partial(_level_to_arg, key='translate_y_pct', default=0.45),
    })
