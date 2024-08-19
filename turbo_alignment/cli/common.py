from pathlib import Path

import typer

from turbo_alignment import pipelines
from turbo_alignment.cli.app import app
from turbo_alignment.common.tf.merge_adapters_to_base import peft_to_base_model
from turbo_alignment.settings.datasets.multimodal import (
    MultimodalDatasetProcessingSettings,
)
from turbo_alignment.settings.pipelines import MergeAdaptersToBaseModelSettings


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


@app.command(name='merge_adapters_to_base', help='Merge peft adapters into base model')
def merge_adapters_to_base_entrypoint(
    settings_path: Path = typer.Option(..., '--settings_path', exists=True, help='Path to script config file')
) -> None:
    settings = MergeAdaptersToBaseModelSettings.parse_file(settings_path)
    peft_to_base_model(settings)
