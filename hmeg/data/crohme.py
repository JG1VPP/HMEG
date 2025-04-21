import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from mmengine.dataset import BaseDataset
from mmengine.fileio import list_dir_or_file
from mmengine.registry import DATASETS
from PIL import Image

from hmeg.model.layout import boxes_to_layout_matrix


def box2layout(boxes, img_size=(256, 256)):
    H, W = img_size
    return boxes_to_layout_matrix(boxes, H=H, W=W)


@DATASETS.register_module()
class CROHME(BaseDataset):
    def __init__(self, npy_path: str, img_path: str, **kwargs):
        self.img_path = Path(img_path)
        self.npy_path = Path(npy_path)
        super().__init__(**kwargs)

    def load_data_list(self):
        files = list_dir_or_file(
            self.npy_path,
            list_dir=False,
            list_file=True,
            suffix=".npy",
            recursive=True,
        )

        return list(map(self.load_file, files))

    def load_file(self, name):
        link = np.load(self.npy_path.joinpath(name), allow_pickle=True).item()
        link.update(img_path=self.img_path.joinpath(name).with_suffix(".png"))
        return link


#        self.transform_64 = T.Compose(
#            [
#                T.Resize((round(image_size[0] / 4), round(image_size[1] / 4))),
#                T.ToTensor(),
#            ]
#        )
#        self.transform_128 = T.Compose(
#            [
#                T.Resize((round(image_size[0] / 2), round(image_size[1] / 2))),
#                T.ToTensor(),
#            ]
#        )
#        self.transform_256 = T.Compose([T.Resize(image_size), T.ToTensor()])
#
#    def __len__(self):
#        return len(self.names)
#
#    def __getitem__(self, index):
#        img_path = self.image_paths[index]
#
#        image = Image.open(img_path).convert("RGB")
#        image_64 = self.transform_64(image)
#        image_128 = self.transform_128(image)
#        image_256 = self.transform_256(image)
#        # image = torch.cat([image] * 3, dim=0)
#        bbox = self.npy[index]["bbox"]
#        edge_type = self.npy[index]["edge_type"]
#
#        objs = bbox[:, 0].long()
#        boxes = bbox[:, 1:]
#
#        n = edge_type.shape[0]
#        triples = []
#        for row in range(n):
#            for col in range(n):
#                triples.append([row, edge_type[row, col], col])
#        triples = torch.LongTensor(triples)
#        # TODO layout gt
#        layout = box2layout(boxes, img_size=(64, 64))
#        return image_64, image_128, image_256, objs, boxes, layout, triples
