from pathlib import Path
import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import list_dir_or_file
from mmengine.registry import DATASETS


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
