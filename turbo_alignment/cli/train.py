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
    ),
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
    ),
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
    ),
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
    ),
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
    ),
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
    ),
) -> None:
    experiment_settings = pipeline_settings.ClassificationTrainExperimentSettings.parse_file(experiment_settings_path)
    pipelines.TrainClassificationStrategy().run(experiment_settings)


@app.command(name='train_multimodal', help='Train Multimodal')
def multimodal_training(
    experiment_settings_path: Path = typer.Option(
        ...,
        '--experiment_settings_path',
        exists=True,
        help='Path to experiment config file',
    ),
) -> None:
    experiment_settings = pipeline_settings.MultimodalTrainExperimentSettings.parse_file(experiment_settings_path)
    pipelines.TrainMultimodalStrategy().run(experiment_settings)


@app.command(name='train_rag', help='Train RAG')
def rag_training(
    experiment_settings_path: Path = typer.Option(
        ...,
        '--experiment_settings_path',
        exists=True,
        help='Path to experiment config file',
    ),
) -> None:
    experiment_settings = pipeline_settings.RAGTrainExperimentSettings.parse_file(experiment_settings_path)
    pipelines.TrainRAGStrategy().run(experiment_settings)


@app.command(name='train_reinforce', help='Train REINFORCE pipeline')
def reinforce_training(
    experiment_settings_path: Path = typer.Option(
        ...,
        '--experiment_settings_path',
        exists=True,
        help='Path to experiment config file',
    ),
) -> None:
    import ray
    from turbo_alignment.trainers.online.ray.rayactor_group import RayGroup
    from turbo_alignment.trainers.online.ray.vllm_engine import create_vllm_engines
    from turbo_alignment.trainers.online.reward_actor import RewardModel
    from turbo_alignment.trainers.online.reference_actor import ReferenceModel
    
    ray.init(address="auto")
    
    experiment_settings = pipeline_settings.REINFORCETrainExperimentSettings.parse_file(experiment_settings_path)

    policy_models = RayGroup(num_nodes=1, num_gpus_per_node=8, ray_actor_type=pipelines.TrainREINFORCEStrategy)
    reward_model = RayGroup(num_nodes=1, num_gpus_per_node=1, ray_actor_type=RewardModel)
    reference_model = RayGroup(num_nodes=1, num_gpus_per_node=1, ray_actor_type=ReferenceModel)

    # TODO_RLOO if possible hide init inside RayGroup
    ray.get(policy_models.async_init_model_from_pretrained())
    ray.get(reward_model.async_init_model_from_pretrained(rm_model=experiment_settings.reward_model_settings.model_path))
    ray.get(reference_model.async_init_model_from_pretrained(pretrain=experiment_settings.model_settings.model_path))

    '''
    TODO_RLOO: 
    1. SEED FIX
    2. PARAMS to REINFORCETrainExperimentSettings
    3. if possible hide creating of vllm engines inside trainer
    '''

    vllm_engines = create_vllm_engines(
        num_engines=experiment_settings.trainer_settings.actor_type.vllm_num_engines,
        tensor_parallel_size=experiment_settings.trainer_settings.actor_type.vllm_tensor_parallel_size,
        pretrain=experiment_settings.model_settings.model_path,
        seed=0,
        enable_prefix_caching=False,
        enforce_eager=False,
        max_model_len=1024,
    )

    ray.get(policy_models.async_fit_actor_model(
        experiment_settings=experiment_settings,
        vllm_engines=vllm_engines,
        reference_model=reference_model, reward_model=reward_model
    ))
