from pathlib import Path

import typer

from turbo_alignment import pipelines
from turbo_alignment.cli.app import app
from turbo_alignment.settings.pipelines.inference.base import (
    InferenceExperimentSettings,
)
from turbo_alignment.settings.pipelines.inference.chat import (
    ChatInferenceExperimentSettings,
)
from turbo_alignment.settings.pipelines.inference.multimodal import (
    MultimodalInferenceExperimentSettings,
)
from turbo_alignment.settings.pipelines.inference.rag import (
    RAGInferenceExperimentSettings,
)


@app.command(name='inference_chat', help='Infer model on chat dataset')
def infer_chat_entrypoint(
    inference_settings_path: Path = typer.Option(
        ..., '--inference_settings_path', exists=True, help='Path to inference config file'
    )
) -> None:
    inference_settings = ChatInferenceExperimentSettings.parse_file(inference_settings_path)
    pipelines.ChatInferenceStrategy().run(inference_settings)


@app.command(name='inference_rm', help='Infer model on rm dataset')
def infer_rm_entrypoint(
    inference_settings_path: Path = typer.Option(
        ..., '--inference_settings_path', exists=True, help='Path to inference config file'
    )
) -> None:
    inference_settings = InferenceExperimentSettings.parse_file(inference_settings_path)
    pipelines.RMInferenceStrategy().run(inference_settings)


@app.command(name='inference_classification', help='Infer model on classification dataset')
def infer_classification_entrypoint(
    inference_settings_path: Path = typer.Option(
        ..., '--inference_settings_path', exists=True, help='Path to inference config file'
    )
) -> None:
    inference_settings = InferenceExperimentSettings.parse_file(inference_settings_path)
    pipelines.ClassificationInferenceStrategy().run(inference_settings)


@app.command(name='inference_multimodal', help='Infer model on chat dataset')
def infer_multimodal_entrypoint(
    inference_settings_path: Path = typer.Option(
        ..., '--inference_settings_path', exists=True, help='Path to inference config file'
    )
) -> None:
    inference_settings = MultimodalInferenceExperimentSettings.parse_file(inference_settings_path)
    pipelines.MultimodalInferenceStrategy().run(inference_settings)


@app.command(name='inference_rag', help='Infer model on classification dataset')
def infer_rag_entrypoint(
    inference_settings_path: Path = typer.Option(
        ..., '--inference_settings_path', exists=True, help='Path to inference config file'
    )
) -> None:
    inference_settings = RAGInferenceExperimentSettings.parse_file(inference_settings_path)
    pipelines.RAGInferenceStrategy().run(inference_settings)
