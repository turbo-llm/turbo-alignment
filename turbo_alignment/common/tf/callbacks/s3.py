import os
import re
from pathlib import Path

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.common.s3.checkpoints_handler import S3CheckpointsHandler
from turbo_alignment.settings.s3 import CheckpointUploaderCallbackParameters

logger = get_project_logger()


class CheckpointUploaderCallback(TrainerCallback):
    def __init__(self, parameters: CheckpointUploaderCallbackParameters, s3_handler: S3CheckpointsHandler) -> None:
        self._parameters = parameters
        self._s3_handler = s3_handler

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        logger.info(f"💾 Saving experiment metadata to {Path(args.output_dir) / 'experiment_metadata.json'}")

        last_checkpoint_directory = get_last_checkpoint(args.output_dir)
        last_checkpoint_basename = str(last_checkpoint_directory).rsplit('/', maxsplit=1)[-1]

        self._s3_handler.upload_local_file(
            target_directory=Path(self._parameters.checkpoints_directory) / last_checkpoint_basename,
            file_path=Path(args.output_dir) / 'experiment_metadata.json',
        )

        return control

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        last_checkpoint_directory = get_last_checkpoint(args.output_dir)
        last_checkpoint_basename = str(last_checkpoint_directory).rsplit('/', maxsplit=1)[-1]

        if last_checkpoint_directory is None:
            return control

        self._upload_checkpoint(checkpoint_directory=Path(last_checkpoint_directory))

        self._s3_handler.upload_local_file(
            target_directory=Path(self._parameters.checkpoints_directory) / last_checkpoint_basename,
            file_path=Path(args.output_dir) / 'experiment.config',
        )

        local_checkpoints_names = self._retrieve_recent_checkpoint_names(output_directory=Path(args.output_dir))

        remote_checkpoints_names = self._s3_handler.list_folders_keys(folder=self._parameters.checkpoints_directory)

        self._remove_old_checkpoints(
            local_checkpoints_names=local_checkpoints_names, remote_checkpoints_names=remote_checkpoints_names
        )

        return control

    def _upload_checkpoint(self, checkpoint_directory: Path) -> None:
        last_checkpoint_basename = str(checkpoint_directory).rsplit('/', maxsplit=1)[-1]
        s3_key = f'{self._parameters.checkpoints_directory}/{last_checkpoint_basename}'
        logger.info(f'🥭 Start to upload checkpoint {checkpoint_directory} to s3 with key {s3_key}')
        self._s3_handler.upload_local_files(
            path=checkpoint_directory,
            target_directory=s3_key,
            save_optimizer_states=self._parameters.save_optimizer_states,
        )
        logger.info(f'🥝 Successfully uploaded checkpoint {checkpoint_directory} to s3 with key {s3_key}')

    @staticmethod
    def _retrieve_recent_checkpoint_names(output_directory: Path) -> list[str]:
        ordering_and_checkpoint_path = []
        glob_checkpoints = [str(x) for x in output_directory.glob('checkpoint-*') if x.is_dir()]
        for path in glob_checkpoints:
            regex_match = re.match('.*(checkpoint-([0-9]+))', path)
            if regex_match is not None and regex_match.groups() is not None:
                ordering_and_checkpoint_path.append((int(regex_match.groups()[1]), regex_match.groups()[0]))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]

        logger.info(f'🍓 Most recent checkpoints are {checkpoints_sorted}')
        return checkpoints_sorted

    def _remove_old_checkpoints(self, local_checkpoints_names: list[str], remote_checkpoints_names: list[str]) -> None:
        for remote_checkpoint_path in remote_checkpoints_names:
            if remote_checkpoint_path not in local_checkpoints_names:
                logger.info(
                    f'''
                            🥭 Start to remove {remote_checkpoint_path} since it is
                            not in {local_checkpoints_names}'''
                )
                self._s3_handler.remove_directory(
                    os.path.join(self._parameters.checkpoints_directory, remote_checkpoint_path)
                )
                logger.info('🥝 Successfully removed %s from s3', remote_checkpoint_path)
