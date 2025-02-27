from typing import Any

from wandb.sdk.lib.disabled import RunDisabled
from wandb.sdk.wandb_run import Run

import wandb
from turbo_alignment.settings.logging.weights_and_biases import WandbSettings


def create_wandb_run(parameters: WandbSettings, config: dict[str, Any] | None = None) -> Run | RunDisabled:
    wandb_run = wandb.init(
        project=parameters.project_name,
        name=parameters.run_name,
        entity=parameters.entity,
        notes=parameters.notes,
        tags=parameters.tags,
        config=config,
        mode=parameters.mode,
    )

    if wandb_run is None:
        raise ValueError('Wandb run is None')

    return wandb_run
