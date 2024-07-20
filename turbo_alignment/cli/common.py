from pathlib import Path

import typer

from turbo_alignment import pipelines
from turbo_alignment.cli.app import app
from turbo_alignment.common.tf.convert_to_base_model import peft_to_base_model
from turbo_alignment.settings.datasets.multimodal import (
    MultimodalDatasetProcessingSettings,
)
from turbo_alignment.settings.pipelines import ConvertToBaseModelSettings


@app.command(
    name='preprocess_multimodal_dataset', help='Read and encode multimodal objects from image / audio dataset'
)
def process_multimodal_dataset(
    settings_path: Path = typer.Option(
        ..., '--settings_path', exists=True, help='Path to multimodal dataset processing settings'
    )
):
    settings = MultimodalDatasetProcessingSettings.parse_file(settings_path)
    pipelines.PreprocessMultimodalDatasetStrategy().run(settings)


@app.command(name='convert_to_base', help='Convert peft adapters to base model format')
def convert_to_base_entrypoint(
    settings_path: Path = typer.Option(..., '--settings_path', exists=True, help='Path to script config file')
) -> None:
    settings = ConvertToBaseModelSettings.parse_file(settings_path)
    peft_to_base_model(settings)
