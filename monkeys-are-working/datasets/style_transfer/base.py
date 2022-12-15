from abc import abstractmethod
from os.path import isdir
from typing import List, Dict

import pandas as pd
import torch

import os, sys

sys.path.append(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
))

from common import read_image

from torch.utils.data import Dataset
import torchvision.transforms as T

from tqdm import tqdm


class StyleTransferDataset(Dataset):

    def __init__(
            self,
            path: str,
            split: str,
            images_height: int = 256,
            images_width: int = 256,
    ):
        assert isdir(path)
        self.path: str = path
        self.split: str = split

        assert isinstance(images_height, int) and images_height >= 1
        assert isinstance(images_width, int) and images_height >= 1
        self.images_height: int = images_height
        self.images_width: int = images_width

        self.metas_df: pd.DataFrame = self._parse_metas()
        self.styles: List[str] = list(self.metas_df["style"].unique())

    def __len__(self):
        return len(self.metas_df)

    def __getitem__(self, i):
        row = self.metas_df.iloc[i]
        item: Dict[str, torch.Tensor] = {
            "style": row["style"],
            "image": torch.from_numpy(
                read_image(row["path"])
            ),
        }
        item["image"] = T.Compose([
            T.Resize(max(self.images_height, self.images_width)),
            T.RandomCrop(max(self.images_height, self.images_width)),
        ])(item["image"])
        return item

    def get_style_indices(self, style: str):
        assert style in self.styles
        return self.metas_df[self.metas_df["style"] == style].index.tolist()

    @abstractmethod
    def _parse_metas(self) -> pd.DataFrame:
        pass
