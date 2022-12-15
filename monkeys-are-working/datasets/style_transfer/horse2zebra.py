from os.path import join, exists
from typing import List, Dict

import pandas as pd
from icecream import ic

from base import StyleTransferDataset


class Horse2ZebraDataset(StyleTransferDataset):

    def __init__(self, split: str, **kwargs):
        assert split in {"train", "test"}
        super().__init__(
            split=split,
            **kwargs
        )

    def _parse_metas(self):
        dataset_metadata = pd.read_csv(join(self.path, "metadata.csv"))
        metas: List[Dict[str, str]] = [
            {
                "path": join(self.path, image_path),
                "style": "zebra" if "Zebra" in style else "horse",
            }
            for image_path, style, split in zip(dataset_metadata["image_path"],
                                                dataset_metadata["domain"],
                                                dataset_metadata["split"])
            if split == self.split and exists(join(self.path, image_path))
        ]
        metas_df = pd.DataFrame(metas)
        return metas_df


if __name__ == "__main__":
    for split in {"test", "train"}:
        dataset = Horse2ZebraDataset(
            path=join("..", "..", "..", "datasets", "vision", "horse2zebra"),
            split=split,
        )
    ic(len(dataset), dataset.styles)
