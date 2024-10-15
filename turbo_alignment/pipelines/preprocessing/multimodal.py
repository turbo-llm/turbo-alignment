from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from accelerate import Accelerator
from turbo_alignment.common.registry import Params
import os
from accelerate.utils import gather_object
from safetensors.torch import save_file
from tqdm import tqdm

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
	def __init__(self, *args, **kwargs):
		self.accelerator = Accelerator()

	def run(self, experiment_settings: MultimodalDatasetProcessingSettings) -> None:
		if self.accelerator.is_main_process:
			logger.info(f'👩 Start dataset processing with the following settings:\n{experiment_settings}')

		reader, encoder = self._load_modality_reader_encoder(
			experiment_settings.reader_settings,
			experiment_settings.encoder_settings,
			experiment_settings.modality,
		)
		self._read_modality_objects(reader, encoder, experiment_settings)

		if self.accelerator.is_main_process:
			logger.info(f'👩 Saved!')

	def _process_function(self, reader, encoder, batch_file_paths, experiment_settings, batch_idx):
		modality_objects = []
		for file_path in batch_file_paths:
			modality_objects.append(reader.read(str(file_path)))
		modality_objects = torch.cat(modality_objects)
		encoded_modality_objects = encoder.encode(modality_objects.to(self.accelerator.device)).detach().cpu()
		safetensors_dict_batch = self._get_safetensor_dict(encoded_modality_objects, batch_file_paths)

		return safetensors_dict_batch

	@staticmethod
	def _save_tensor(tensor, filename, experiment_settings):
		logger.info(f'saving {filepath}', tensor.shape)
		filepath = experiment_settings.output_file_path / (
				filename
				+ '.'
				+ experiment_settings.modality.value
				+ '.'
				+ experiment_settings.encoder_settings.modality_encoder_type
				+ '.pt'
			)
		torch.save(tensor, filepath)
	
	def _process_files(self, reader, encoder, files_paths, experiment_settings):
		batches_all = np.array_split(files_paths, len(files_paths) // experiment_settings.batch_size)

		for i, batch in enumerate(tqdm(batches_all)):
			try:
				logger.info(f'📖 Processing batch {i} / {len(batches_all)}')
				batch_output = self._process_function(reader, encoder, batch, experiment_settings, i)
				torch.save(batch_output, experiment_settings.output_file_path / (
						'batch_' + str(i) + '.' + experiment_settings.modality.value
						+ '.'
						+ experiment_settings.encoder_settings.modality_encoder_type
						+ '.pt'
					)
				)
			except Exception as exc:
				logger.error(f'Error reading file: {exc}')

	def _async_process_files(self, reader, encoder, files_paths, experiment_settings):
		logger.info(f'👩 Processing with accelerate!')
		batches_all = np.array_split(files_paths, len(files_paths) // experiment_settings.batch_size)

		self.accelerator.wait_for_everyone()

		with self.accelerator.split_between_processes(batches_all) as batches:
			for i, batch in enumerate(tqdm(batches)):
				try:
					logger.info(f'📖 Encoding batch {i} / {len(batches)}')
					batch_output = self._process_function(reader, encoder, batch, experiment_settings, i)
					torch.save(batch_output, experiment_settings.output_file_path / (
							'process_' + str(self.accelerator.process_index) + 'batch_' + str(i) + '.' + experiment_settings.modality.value
							+ '.'
							+ experiment_settings.encoder_settings.modality_encoder_type
							+ '.pt'
						)
					)

				except Exception as exc:
					logger.error(f'Error reading file: {exc}')

	def _load_modality_reader_encoder(
		self,
		reader_settings: ModalityReaderSettings,
		encoder_settings: ModalityEncoderSettings,
		modality: Modality,
	) -> Tuple[BaseModalityReader, BaseModalityEncoder]:
		device = self.accelerator.device
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

		if self.accelerator.is_main_process:
			logger.info('📖 Reading modality objects...')
		files_paths: list[Path] = []
		for extension in available_extensions:
			files_paths.extend(experiment_settings.dataset_path.glob(f'*.{extension}'))

		if os.environ.get('ACCELERATE_ENABLED', 'false') == 'true':
			self._async_process_files(reader, encoder, files_paths, experiment_settings)
		else:
			self._process_files(reader, encoder, files_paths, experiment_settings)

	@staticmethod
	def _get_safetensor_dict(encoded_modality_tensors, encoded_file_paths):
		tensors = {}
		for file, tensor in zip(encoded_file_paths, encoded_modality_tensors):
			tensors[file.name] = tensor.detach()
		return tensors
