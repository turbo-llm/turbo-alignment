from pathlib import Path

import typer

from turbo_alignment import pipelines
from turbo_alignment.cli.app import app
from turbo_alignment.settings import pipelines as pipeline_settings


@app.command(name='train_sft', help='Run PEFT pipeline')
def train_sft_entrypoint(
    experiment_settings_path: Path = typer.Option(
        ...,
        '--experiment_settings_path',
        exists=True,
        help='Path to experiment config file',
    )
) -> None:
    experiment_settings = pipeline_settings.SftTrainExperimentSettings.parse_file(experiment_settings_path)
    pipelines.TrainSFTStrategy().run(experiment_settings)


def set_prct():
    import prctl
    prctl.set_ptracer(prctl.SET_PTRACER_ANY)


@app.command(name='train_dpo', help='Run DPO pipeline')
def train_dpo_entrypoint(
    experiment_settings_path: Path = typer.Option(
        ...,
        '--experiment_settings_path',
        exists=True,
        help='Path to experiment config file',
    )
) -> None:
    import prctl
    prctl.set_ptracer(prctl.SET_PTRACER_ANY)

    import os
    os.register_at_fork(after_in_child=set_prct)

    experiment_settings = pipeline_settings.DPOTrainExperimentSettings.parse_file(experiment_settings_path)
    pipelines.TrainDPOStrategy().run(experiment_settings)


@app.command(name='train_rm', help='Run RM pipeline')
def train_rm_entrypoint(
    experiment_settings_path: Path = typer.Option(
        ...,
        '--experiment_settings_path',
        exists=True,
        help='Path to experiment config file',
    )
) -> None:
    experiment_settings = pipeline_settings.RMTrainExperimentSettings.parse_file(experiment_settings_path)
    pipelines.TrainRMStrategy().run(experiment_settings)


@app.command(name='train_classification', help='Train Classifier')
def classification_training(
    experiment_settings_path: Path = typer.Option(
        ...,
        '--experiment_settings_path',
        exists=True,
        help='Path to experiment config file',
    )
) -> None:
    experiment_settings = pipeline_settings.ClassificationTrainExperimentSettings.parse_file(experiment_settings_path)
    pipelines.TrainClassificationStrategy().run(experiment_settings)
