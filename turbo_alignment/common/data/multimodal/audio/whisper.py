from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoFeatureExtractor

from turbo_alignment.common.data.multimodal.image.base import BaseImageReader
from turbo_alignment.common.data.multimodal.registry import AudioModalityReaderRegistry
from turbo_alignment.settings.modality import ModalityReader


@AudioModalityReaderRegistry.register(ModalityReader.WHISPER)
class WhisperAudioReader(BaseImageReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained('openai/whisper-large-v3')

    def read(self, path: str) -> torch.Tensor:
        data, _ = sf.read(path)
        inputs = self.whisper_feature_extractor(
            data, sampling_rate=self.whisper_feature_extractor.sampling_rate, return_tensors='pt'
        )
        return inputs['input_features'][0]
