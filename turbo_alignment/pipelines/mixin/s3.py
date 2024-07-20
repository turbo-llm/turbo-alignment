from turbo_alignment.common.s3.checkpoints_handler import S3CheckpointsHandler
from turbo_alignment.common.tf.callbacks import CheckpointUploaderCallback
from turbo_alignment.settings.s3 import (
    CheckpointUploaderCallbackParameters,
    S3HandlerParameters,
)


class S3Mixin:
    @staticmethod
    def _get_s3_handler(s3_handler_parameters: S3HandlerParameters) -> S3CheckpointsHandler:
        return S3CheckpointsHandler(parameters=s3_handler_parameters)

    @staticmethod
    def _get_checkpoint_uploader_callback(
        s3_handler: S3CheckpointsHandler, checkpoint_uploader_callback_parameters: CheckpointUploaderCallbackParameters
    ) -> CheckpointUploaderCallback:
        checkpoint_uploader_callback = CheckpointUploaderCallback(
            parameters=checkpoint_uploader_callback_parameters, s3_handler=s3_handler
        )
        if s3_handler.directory_exists(directory_path=checkpoint_uploader_callback_parameters.directory):
            raise ValueError(f'🍅 Directory {checkpoint_uploader_callback_parameters.directory} exists in bucket')

        return checkpoint_uploader_callback
