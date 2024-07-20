import os
import pathlib
from pathlib import Path

import boto3

from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.settings.s3 import S3HandlerParameters

logger = get_project_logger()


class S3CheckpointsHandler:
    def __init__(self, parameters: S3HandlerParameters) -> None:
        self._resource = boto3.resource(
            service_name='s3',
            endpoint_url=parameters.host,
            aws_access_key_id=parameters.aws_access_key_id,
            aws_secret_access_key=parameters.aws_secret_access_key,
        )

        self._parameters = parameters

    def upload_local_files(self, path: pathlib.Path, target_directory: str, save_optimizer_states: bool) -> None:
        file_names_count = len(list(os.walk(path)))

        for root, _, files in os.walk(path):
            for index, file in enumerate(files):
                if not save_optimizer_states and 'global_step' in root:
                    continue

                root_path = Path(root)
                file_path = root_path / file

                target_directory_path = Path(target_directory)

                key = target_directory_path / file_path.relative_to(path)

                logger.info(
                    f'''🥭 Start to upload file {index} / {file_names_count}
                                    from local {file} to remote s3 with key {key}'''
                )

                self._resource.meta.client.upload_file(
                    Filename=str(file_path), Bucket=self._parameters.bucket, Key=str(key)
                )

                logger.info(
                    f'''🥝 Successfully uploaded file {index} / {file_names_count}
                                    from local {file_path} to remote s3 with key {key}'''
                )

    def upload_local_file(self, file_path: Path, target_directory: Path) -> None:
        self._resource.meta.client.upload_file(
            Filename=str(file_path), Bucket=self._parameters.bucket, Key=str(target_directory / file_path.name)
        )

    def directory_exists(self, directory_path: Path) -> bool:
        objects = self._resource.Bucket(self._parameters.bucket).objects.filter(Prefix=f'{directory_path}/')
        return len([o.key for o in objects]) > 0

    def remove_directory(self, prefix: str) -> None:
        logger.info(f'🥭 Start to remove directory with prefix {prefix}')
        bucket = self._resource.Bucket(self._parameters.bucket)
        bucket.objects.filter(Prefix=prefix).delete()
        logger.info(f'🥝 Successfully removed directory with prefix {prefix}')

    def list_folders_keys(self, folder: str) -> list[str]:
        folders = []

        result = self._resource.meta.client.list_objects(
            Bucket=self._parameters.bucket,
            Prefix=f'{folder}/',
            Delimiter='/',
            MaxKeys=1_000,
        )

        prefixes = result.get('CommonPrefixes', [])
        for prefix in prefixes:
            _folder = prefix['Prefix'].split('/')
            # folder always ends in / so -1 will
            # point to empty string
            right_most_folder_name = _folder[-2]
            folders.append(right_most_folder_name)

        return folders
