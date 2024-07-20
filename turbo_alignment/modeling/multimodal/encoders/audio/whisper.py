import torch
from transformers import WhisperForConditionalGeneration

from turbo_alignment.modeling.multimodal.encoders.audio.base import BaseAudioEncoder
from turbo_alignment.modeling.multimodal.encoders.registry import (
    ModalityEncoderRegistry,
)
from turbo_alignment.settings.modality import ModalityEncoderType


@ModalityEncoderRegistry.register(ModalityEncoderType.WHISPER)
class WhisperAudioModeling(BaseAudioEncoder):
    def __init__(self, model_whisper: WhisperForConditionalGeneration | None = None):
        super().__init__()

        if model_whisper is not None:
            self.model_whisper = model_whisper
        else:
            self.model_whisper = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3')
        self.whisper_encoder = self.model_whisper.model.encoder

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        audio_embeddings = self.whisper_encoder(inputs).last_hidden_state
        return audio_embeddings.mean(dim=1)

    @property
    def emb_dim(self) -> int:
        return self.whisper_encoder.config.d_model
