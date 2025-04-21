import argparse
import itertools
import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from mmengine import Config
from mmengine.runner import Runner
from torch.utils.data import DataLoader
from tqdm import tqdm

from hmeg.data import build_dataset
from hmeg.model.discriminators import AcCropDiscriminator, PatchDiscriminator
from hmeg.model.generator import Sg2ImModel
from hmeg.model.losses import get_gan_losses
from hmeg.model.metrics import jaccard

if __name__ == "__main__":
    config = Config.fromfile("config.py")

    train = build_dataset(config.train)

    #    train_dataloader = Runner.build_dataloader(dict(
    #        batch_size=8,
    #        num_workers=8,
    #        persistent_workers=True,
    #        sampler=dict(type='DefaultSampler', shuffle=True),
    #        dataset=train,
    #    ))

    print(len(train))
