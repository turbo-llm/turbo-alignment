from pathlib import Path

import typer

import turbo_alignment.pipelines.sampling as sampling_pipelines
from turbo_alignment.cli.app import app
from turbo_alignment.settings.pipelines.sampling import (
    RandomSamplingSettings,
    RSOSamplingSettings,
    SamplingWithRMSettings,
)


@app.command(name='rso_sample', help='RSO sampling')
def rso_sampling_entrypoint(
    experiment_settings_path: Path = typer.Option(
        ..., '--experiment_settings_path', exists=True, help='Path to settings config file'
    )
) -> None:
    experiment_settings = RSOSamplingSettings.parse_file(experiment_settings_path)
    sampling_pipelines.RSOSamplingStrategy().run(experiment_settings)


@app.command(name='random_sample', help='Random sampling')
def random_sampling_entrypoint(
    experiment_settings_path: Path = typer.Option(
        ..., '--experiment_settings_path', exists=True, help='Path to settings config file'
    )
) -> None:
    experiment_settings = RandomSamplingSettings.parse_file(experiment_settings_path)
    sampling_pipelines.RandomSamplingStrategy().run(experiment_settings)


@app.command(name='rm_sample', help='Reward Modelsampling')
def rm_sampling_entrypoint(
    experiment_settings_path: Path = typer.Option(
        ..., '--experiment_settings_path', exists=True, help='Path to settings config file'
    )
) -> None:
    experiment_settings = SamplingWithRMSettings.parse_file(experiment_settings_path)
    sampling_pipelines.SamplingStrategyWithRM().run(experiment_settings)
