from mmcv import TRANSFORMS  # noqa
from mmengine import Config
from mmengine.runner import Runner

import hmeg  # noqa

if __name__ == "__main__":
    config = Config.fromfile("config.py")
    runner = Runner.from_cfg(config)
    print(runner)
