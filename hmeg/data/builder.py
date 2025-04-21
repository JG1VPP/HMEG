from mmengine import DATASETS, build_from_cfg
from mmcv import TRANSFORMS  # noqa


def build_dataset(config):
    return build_from_cfg(config, DATASETS)
