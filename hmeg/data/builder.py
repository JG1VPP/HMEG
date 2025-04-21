from mmcv import DATASETS, build_from_cfg


def build_dataset(config):
    return build_from_cfg(config, DATASETS)
