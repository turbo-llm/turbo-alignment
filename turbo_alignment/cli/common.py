from pathlib import Path

import typer

from turbo_alignment.cli.app import app
from turbo_alignment.common.tf.merge_adapters_to_base import peft_to_base_model
from turbo_alignment.settings.pipelines import MergeAdaptersToBaseModelSettings


@app.command(name='merge_adapters_to_base', help='Merge peft adapters into base model')
def merge_adapters_to_base_entrypoint(
    settings_path: Path = typer.Option(..., '--settings_path', exists=True, help='Path to script config file')
) -> None:
    settings = MergeAdaptersToBaseModelSettings.parse_file(settings_path)
    peft_to_base_model(settings)
