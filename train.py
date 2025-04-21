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
from torch.utils.data import DataLoader
from tqdm import tqdm

from hmeg.data import imagenet_deprocess_batch
from hmeg.data.crohme import CROHMELabelGraphDataset, crohme_collate_fn
from hmeg.model.discriminators import AcCropDiscriminator, PatchDiscriminator
from hmeg.model.generator import Sg2ImModel
from hmeg.model.losses import get_gan_losses
from hmeg.model.metrics import jaccard

if __name__ == "__main__":
    CROHME_DIR = os.path.expanduser("datasets/crohme2019")
    
    with open(os.path.join(CROHME_DIR, "vocab.json")) as f:
        vocab = json.load(f)
    nc = len(vocab["object_idx_to_name"])
    npy_dir = os.path.join(CROHME_DIR, "link_npy")
    names = [name[:-4] for name in os.listdir(npy_dir)]

    train = CROHMELabelGraphDataset(CROHME_DIR, names[:-500], nc=nc, image_size=(256, 256))
    valid = CROHMELabelGraphDataset(CROHME_DIR, names[-500:], nc=nc, image_size=(256, 256))
    
    print(len(train))
