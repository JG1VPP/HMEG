from mmcv import TRANSFORMS  # noqa
from mmengine import Config
from mmengine.runner import Runner

from hmeg.data import CROHME  # noqa

if __name__ == "__main__":
    config = Config.fromfile("config.py")
    train = Runner.build_dataloader(config.train)
    print(len(train))
    print(next(iter(train)))
