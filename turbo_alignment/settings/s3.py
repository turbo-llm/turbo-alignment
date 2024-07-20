from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings


class S3HandlerParametersWrongBucketException(Exception):
    ...


class S3HandlerParameters(BaseSettings):
    host: str
    bucket: str
    aws_access_key_id: str
    aws_secret_access_key: str

    @field_validator('bucket')
    def bucket_name_biglm_is_not_allowed(cls, values: str) -> str:
        if values == 'biglm':
            raise S3HandlerParametersWrongBucketException('Usage of biglm bucket is not allowed')

        return values

    class Config:
        env_file: str = '.env'
        env_prefix: str = 'S3_CHECKPOINTS_'


class CheckpointUploaderCallbackParameters(BaseSettings):
    directory: Path

    save_optimizer_states: bool = True
    tokenizer_folder_name: str = 'tokenizer'
    checkpoints_folder_name: str = 'checkpoints'

    @property
    def tokenizer_directory(self) -> str:
        return f'{self.directory}/{self.tokenizer_folder_name}'

    @property
    def checkpoints_directory(self) -> str:
        return f'{self.directory}/{self.checkpoints_folder_name}'

    class Config:
        env_prefix: str = 'CHECKPOINT_UPLOADER_CALLBACK_'


class ExperimentMetadata(BaseSettings):
    wandb_run_id: str = 'wandb_run_id'
    platform_job_name: str = 'platform_job_name'
    last_git_commit_hash: str = 'last_git_commit_hash'
