from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import torch
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
        reader_settings: ModalityReaderSettings, encoder_settings: ModalityEncoderSettings, modality: Modality
    ) -> Tuple[BaseModalityReader, BaseModalityEncoder]:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        reader = ModalityReaderRegistry.by_name(modality).from_params(
            Params({'type': reader_settings.reader_type, 'reader_path': reader_settings.reader_path})
        )
        encoder = ModalityEncoderRegistry.by_name(encoder_settings.modality_encoder_type)(
            encoder_path=encoder_settings.encoder_path
        ).to(device)
        return (reader, encoder)

    def _read_modality_objects(self, reader: BaseModalityReader, dataset_path: Path):
        modality_tensors = []

        available_extensions = ('jpg', 'jpeg', 'png', 'svg')

        total_number_of_objects = len(list(dataset_path.iterdir()))

        logger.info('📖 Reading modality objects...')
        files_paths: list[Path] = []
        for extension in available_extensions:
            files_paths.extend(dataset_path.glob(f'*.{extension}'))

        modality_tensors = self._async_process_files(reader, files_paths, total_number_of_objects)

        modality_tensors = torch.cat(modality_tensors)
        return modality_tensors, files_paths

    @staticmethod
    def _encode_modality_objects(
        modality_tensors: torch.Tensor, encoder: BaseModalityEncoder, batch_size: int
    ) -> torch.Tensor:
        encoded_modality_tensors = []

        logger.info('👩‍💻 Encoding objects...')
        batched_modality_tensors = modality_tensors.split(batch_size)
        for i, batch in enumerate(batched_modality_tensors):
            logger.info(f'👩‍💻 Encoded {i} / {len(batched_modality_tensors)} batches')
            encoded_modality_tensor_batch = encoder.encode(batch.to(encoder.device)).detach().cpu()
            encoded_modality_tensors.append(encoded_modality_tensor_batch)

        encoded_modality_tensors = torch.cat(encoded_modality_tensors)

        return encoded_modality_tensors

    @staticmethod
    def _build_encoder_config(encoder) -> dict:
        return {'emb_dim': encoder.emb_dim}

    @staticmethod
    def _async_process_files(reader, files_paths, total_number_of_objects):
        modality_tensors = [None] * len(files_paths)

        with ThreadPoolExecutor() as executor:
            future_to_index = {
                executor.submit(reader.read, str(file_path)): i for i, file_path in enumerate(files_paths)
            }

            for i, future in enumerate(as_completed(future_to_index)):
                index = future_to_index[future]
                if i % 1000 == 0:
                    logger.info(f'📖 Read {i} / {total_number_of_objects} objects')
                try:
                    modality_tensor = future.result()
                    modality_tensors[index] = modality_tensor
                except Exception as exc:
                    logger.error(f'Error reading file at index {index}: {exc}')

        logger.info(f'📖 Successfully read {len(files_paths)} objects!')
        return modality_tensors

    @staticmethod
    def _get_safetensor_dict(encoded_modality_tensors, encoded_file_paths):
        tensors = {}
        for file, tensor in zip(encoded_file_paths, encoded_modality_tensors):
            tensors[file.name] = tensor.clone().detach()
        return tensors

    def run(self, experiment_settings: MultimodalDatasetProcessingSettings) -> None:
        logger.info(f'👩 Start dataset processing with the following settings:\n{experiment_settings}')
        reader, encoder = self._load_modality_reader_encoder(
            experiment_settings.reader_settings, experiment_settings.encoder_settings, experiment_settings.modality
        )
        modality_tensors, encoded_file_paths = self._read_modality_objects(reader, experiment_settings.dataset_path)
        encoded_modality_tensors = self._encode_modality_objects(
            modality_tensors, encoder, experiment_settings.batch_size
        )

        logger.info('👩 Saving encoded objects...')

        experiment_settings.output_file_path.mkdir(parents=True, exist_ok=True)

        tensors = self._get_safetensor_dict(encoded_modality_tensors, encoded_file_paths)

        del encoded_modality_tensors

        save_file(
            tensors,
            experiment_settings.output_file_path
            / (
                experiment_settings.modality.value
                + '.'
                + experiment_settings.encoder_settings.modality_encoder_type
                + '.safetensors'
            ),
        )

        encoder_config_path = experiment_settings.output_file_path / (
            experiment_settings.modality.value
            + '.'
            + experiment_settings.encoder_settings.modality_encoder_type
            + '.config'
        )

        logger.info(f'👩 Saving encoder config to {encoder_config_path}')
        encoder_config = self._build_encoder_config(encoder)
        write_json(encoder_config, encoder_config_path)
