import torch
from mmcv import TRANSFORMS  # noqa
from mmengine import MODELS, Config
from mmengine.optim import build_optim_wrapper
from mmengine.runner import Runner

import hmeg  # noqa

if __name__ == "__main__":
    config = Config.fromfile("config.py")
    runner = Runner.from_cfg(config)
    ckpt = torch.load(config.ckpt)
    runner.model.gen.load_state_dict(ckpt["model_state"])
    runner.train()
