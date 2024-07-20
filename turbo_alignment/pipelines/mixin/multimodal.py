import torch

from turbo_alignment.modeling.multimodal.encoders import ModalityEncoderRegistry
from turbo_alignment.modeling.multimodal.encoders.base import BaseModalityEncoder
from turbo_alignment.settings.modality import Modality, ModalityEncoderSettings


class MultimodalMixin:
    @staticmethod
    def _load_modality_encoders(
        modality_encoder_settings_mapping: dict[Modality, ModalityEncoderSettings | None],
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[Modality, BaseModalityEncoder]:
        encoders_dict: dict[Modality, BaseModalityEncoder] = {}

        for modality, encoder_settings in modality_encoder_settings_mapping.items():
            if encoder_settings:
                encoder = ModalityEncoderRegistry.by_name(encoder_settings.modality_encoder_type)(
                    encoder_path=encoder_settings.encoder_path,
                    is_pickle=encoder_settings.is_pickle,
                )
                encoder = encoder.to(device)
                encoder = encoder.to(dtype)
                encoders_dict[modality] = encoder

        return encoders_dict
