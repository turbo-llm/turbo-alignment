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


@app.command(name='train_dpo', help='Run DPO pipeline')
def train_dpo_entrypoint(
    experiment_settings_path: Path = typer.Option(
        ...,
        '--experiment_settings_path',
        exists=True,
        help='Path to experiment config file',
    )
) -> None:
    experiment_settings = pipeline_settings.DPOTrainExperimentSettings.parse_file(experiment_settings_path)
    pipelines.TrainDPOStrategy().run(experiment_settings)


@app.command(name='train_kto', help='Run KTO pipeline')
def train_kto_entrypoint(
    experiment_settings_path: Path = typer.Option(
        ...,
        '--experiment_settings_path',
        exists=True,
        help='Path to experiment config file',
    )
) -> None:
    experiment_settings = pipeline_settings.KTOTrainExperimentSettings.parse_file(experiment_settings_path)
    pipelines.TrainKTOStrategy().run(experiment_settings)


@app.command(name='train_ddpo', help='Run DDPO pipeline')
def train_ddpo_entrypoint(
    experiment_settings_path: Path = typer.Option(
        ...,
        '--experiment_settings_path',
        exists=True,
        help='Path to experiment config file',
    )
) -> None:
    experiment_settings = pipeline_settings.DDPOTrainExperimentSettings.parse_file(experiment_settings_path)
    pipelines.TrainDDPOStrategy().run(experiment_settings)


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


@app.command(name='train_multimodal', help='Train Multimodal')
def multimodal_training(
    experiment_settings_path: Path = typer.Option(
        ..., '--experiment_settings_path', exists=True, help='Path to experiment config file'
    )
) -> None:
    experiment_settings = pipeline_settings.MultimodalTrainExperimentSettings.parse_file(experiment_settings_path)
    pipelines.TrainMultimodalStrategy().run(experiment_settings)


@app.command(name='train_rag', help='Train RAG')
def rag_training(
    experiment_settings_path: Path = typer.Option(
        ..., '--experiment_settings_path', exists=True, help='Path to experiment config file'
    )
) -> None:
    experiment_settings = pipeline_settings.RAGTrainExperimentSettings.parse_file(experiment_settings_path)
    pipelines.TrainRAGStrategy().run(experiment_settings)
