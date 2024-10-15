from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from accelerate import Accelerator
from turbo_alignment.common.registry import Params
from safetensors.torch import save_file

from turbo_alignment.common.data.io import write_json
from turbo_alignment.common.data.multimodal.base import BaseModalityReader
from turbo_alignment.common.data.multimodal.registry import ModalityReaderRegistry
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.common.registry import Params
from turbo_alignment.modeling.multimodal.encoders import ModalityEncoderRegistry
from turbo_alignment.modeling.multimodal.encoders.base import BaseModalityEncoder
from turbo_alignment.pipelines.base import BaseStrategy
from turbo_alignment.settings.datasets.multimodal import (
    MultimodalDatasetProcessingSettings,
)
from turbo_alignment.settings.modality import (
    Modality,
    ModalityEncoderSettings,
    ModalityReaderSettings,
)

logger = get_project_logger()


class PreprocessMultimodalDatasetStrategy(BaseStrategy):
    @staticmethod
    def _load_modality_reader_encoder(
        reader_settings: ModalityReaderSettings,
        encoder_settings: ModalityEncoderSettings,
        modality: Modality,
        accelerator,
    ) -> Tuple[BaseModalityReader, BaseModalityEncoder]:
        device = accelerator.device
        reader = ModalityReaderRegistry.by_name(modality).from_params(
            Params({'type': reader_settings.reader_type, 'reader_path': reader_settings.reader_path})
        )
        encoder = ModalityEncoderRegistry.by_name(encoder_settings.modality_encoder_type)(
            encoder_path=encoder_settings.encoder_path
        ).to(device)
        return (reader, encoder)

    def _read_modality_objects(self, reader, encoder, experiment_settings):
        modality_tensors = []

        available_extensions = ('jpg', 'jpeg', 'png', 'svg')

        logger.info('📖 Reading modality objects...')
        files_paths: list[Path] = []
        for extension in available_extensions:
            files_paths.extend(experiment_settings.dataset_path.glob(f'*.{extension}'))

        safetensors_full_dict = self._async_process_files(reader, encoder, files_paths, experiment_settings)
        return safetensors_full_dict

    @staticmethod
    def _build_encoder_config(encoder) -> dict:
        return {'emb_dim': encoder.emb_dim}

    def _process_function(self, reader, encoder, batch_file_paths, experiment_settings, batch_idx):
        logger.info(f'📖 Processing batch {batch_idx}')
        modality_objects = []
        for file_path in batch_file_paths:
            modality_objects.append(reader.read(str(file_path)))
        modality_objects = torch.cat(modality_objects)
        encoded_modality_objects = encoder.encode(modality_objects).detach().cpu()
        safetensors_dict_batch = self._get_safetensor_dict(encoded_modality_objects, batch_file_paths)
        logger.info(f'📖 Finish with batch {batch_idx}')

        return safetensors_dict_batch

    def _async_process_files(self, reader, encoder, files_paths, experiment_settings):
        modality_tensors = [None] * len(files_paths)

        batches = np.array_split(files_paths, len(files_paths) // experiment_settings.batch_size)
        safetensors_full_dict = {}

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._process_function, reader, encoder, batch, experiment_settings, i): i
                for i, batch in enumerate(batches)
            }
            for i, future in enumerate(as_completed(futures)):
                try:
                    safetensors_full_dict.update(future.result())
                except Exception as exc:
                    logger.error(f'Error reading file: {exc}')
        return safetensors_full_dict

    @staticmethod
    def _get_safetensor_dict(encoded_modality_tensors, encoded_file_paths):
        tensors = {}
        for file, tensor in zip(encoded_file_paths, encoded_modality_tensors):
            tensors[file.name] = tensor.detach()
        return tensors

    def run(self, experiment_settings: MultimodalDatasetProcessingSettings) -> None:
        logger.info(f'👩 Start dataset processing with the following settings:\n{experiment_settings}')
        accelerator = Accelerator()

        reader, encoder = self._load_modality_reader_encoder(
            experiment_settings.reader_settings,
            experiment_settings.encoder_settings,
            experiment_settings.modality,
            accelerator,
        )
        safetensors_full_dict = self._read_modality_objects(reader, encoder, experiment_settings)

        logger.info(f'👩 Saving safetensors file...')
        save_file(
            safetensors_full_dict,
            experiment_settings.output_file_path
            / (
                experiment_settings.modality.value
                + '.'
                + experiment_settings.encoder_settings.modality_encoder_type
                + '.safetensors'
            ),
        )
        logger.info(f'👩 Saved!')
