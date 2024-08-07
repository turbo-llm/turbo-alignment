# pylint: disable=unused-import
import torch

from turbo_alignment.common.singleton import ParametrizedSingleton
from turbo_alignment.modeling.imagebind.heads.registry import Heads
from turbo_alignment.modeling.imagebind.models import (
    ImageBindArchitectureSettings,
    ImageBindSettings,
)
from turbo_alignment.modeling.imagebind.postprocessors.registry import Postprocessors
from turbo_alignment.modeling.imagebind.preprocessors.registry import Preprocessors
from turbo_alignment.modeling.imagebind.trunks.registry import Trunks


class ImageBindModel(torch.nn.Module):
    def __init__(self, settings: ImageBindArchitectureSettings):
        super().__init__()

        self.modality_heads = Heads(settings)
        self.modality_trunks = Trunks(settings)
        self.modality_preprocessors = Preprocessors(settings)
        self.modality_postprocessors = Postprocessors(settings)

    def forward(self, inputs):
        outputs = {}
        for modality_key, modality_value in inputs.items():
            reduce_list = modality_value.ndim >= 5  # Audio and Video inputs consist of multiple clips
            if reduce_list:
                B, S = modality_value.shape[:2]
                modality_value = modality_value.reshape(B * S, *modality_value.shape[2:])

            if modality_value is not None:
                modality_value = self.modality_preprocessors[modality_key](**{modality_key: modality_value})
                trunk_inputs = modality_value['trunk']
                head_inputs = modality_value['head']
                modality_value = self.modality_trunks[modality_key](**trunk_inputs)
                modality_value = self.modality_heads[modality_key](modality_value, **head_inputs)
                modality_value = self.modality_postprocessors[modality_key](modality_value)

                if reduce_list:
                    modality_value = modality_value.reshape(B, S, -1)
                    modality_value = modality_value.mean(dim=1)

                outputs[modality_key] = modality_value

        return outputs


def load_imagebind(settings: ImageBindSettings) -> ImageBindModel:
    model = ImageBindModel(settings.architecture_settings)

    if settings.weights_path:
        model.load_state_dict(torch.load(settings.weights_path), strict=False)

    if not settings.is_trainable:
        for param in model.parameters():
            param.requires_grad = False

    return model


class ImageBindSingleton(metaclass=ParametrizedSingleton[ImageBindSettings]):  # type: ignore[misc]
    def __init__(self, settings: ImageBindSettings) -> None:
        self._model = load_imagebind(settings)

    def get(self) -> ImageBindModel:
        return self._model
