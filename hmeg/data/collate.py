import torch

from mmengine import FUNCTIONS


@FUNCTIONS.register_module()
def crohme_collate_fn(batch):
    """
    Collate function to be used when wrapping a VgSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving categories for all objects
    - boxes: FloatTensor of shape (O, 4) giving boxes for all objects
    - triples: FloatTensor of shape (T, 3) giving all triples, where
      triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
    - obj_to_img: LongTensor of shape (O,) mapping objects to images;
      obj_to_img[i] = n means that objs[i] belongs to imgs[n]
    - triple_to_img: LongTensor of shape (T,) mapping triples to images;
      triple_to_img[t] = n means that triples[t] belongs to imgs[n].
    """
    all_imgs = []
    all_objs = []
    all_bbox = []
    all_triples = []
    all_layouts = []

    all_object_to_img = []
    all_triple_to_img = []
    
    obj_offset = 0
    
    for n, sample in enumerate(batch):
        all_imgs.append(sample['img'])

        objs = sample['objs']
        bbox = sample['bbox']

        triple = sample['triple'].clone()
        layout = sample['layout']

        triple[:, 0::2] += obj_offset

        all_objs.append(objs)
        all_bbox.append(bbox)
        
        all_layouts.append(layout)
        all_triples.append(triple)

        all_object_to_img.append(torch.zeros(len(objs)).fill_(n))
        all_triple_to_img.append(torch.zeros(len(triple)).fill_(n))

        obj_offset += len(objs)

    return dict(
        imgs=torch.cat(all_imgs),
        objs=torch.cat(all_objs),
        bbox=torch.cat(all_bbox),
        triples=torch.cat(all_triples),
        layouts=torch.cat(all_layouts),
        object_to_img=torch.cat(all_object_to_img),
        triple_to_img=torch.cat(all_triple_to_img),
    )
