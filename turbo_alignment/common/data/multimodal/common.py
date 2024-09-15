from pathlib import Path
import json
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc 
from functools import reduce
import os


import torch

from turbo_alignment.common.data.multimodal.image.base import BaseImageReader
from turbo_alignment.common.data.multimodal.registry import (
    AudioModalityReaderRegistry,
    ImageModalityReaderRegistry,
)
from turbo_alignment.settings.modality import ModalityReader


@AudioModalityReaderRegistry.register(ModalityReader.PICKLE)
@ImageModalityReaderRegistry.register(ModalityReader.PICKLE)
class FileReader(BaseImageReader):
    def __init__(self, **_kwargs):
        self.processed_batches = None
        self.index = None

    def read(self, path: str):
        with open("/from_s3/dataset/llava_next_data_dialogs_index_file/index.json", 'r') as f:
            self.index = json.load(f)
        cur_file = self.index[Path(path).name]
        cur_tensor = torch.load(cur_file)
        special_vector = cur_tensor[Path(path).name].clone()
        del cur_tensor
        del self.index
        gc.collect()
        return special_vector
