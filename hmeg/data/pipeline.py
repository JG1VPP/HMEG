import torch
from mmcv.transforms import BaseTransform, TRANSFORMS
from hmeg.model.layout import boxes_to_layout_matrix


@TRANSFORMS.register_module()
class LabelGraph(BaseTransform):
    def __init__(self, h: int, w: int):
        super().__init__()
        self.h = h
        self.w = w

    def transform(self, results: dict):
        bbox = results["bbox"]
        edge = results["edge_type"]
        objs = bbox[:, 0].long()
        bbox = bbox[:, 1:]
        tups = []
        for row in range(len(edge)):
            for col in range(len(edge)):
                tups.append([row, edge[row, col], col])
        triple = torch.tensor(tups).long()
        layout = boxes_to_layout_matrix(bbox, H=self.h, W=self.w)
        results.update(objs=objs, bbox=bbox, triple=triple, layout=layout)
        return results
