from typing import Any

from clearml import Task

from turbo_alignment.settings.logging.clearml import ClearMLSettings


def create_clearml_task(parameters: ClearMLSettings, config: dict[str, Any] | None = None) -> Task:
    clearml_task = Task.init(
        task_name=parameters.task_name, project_name=parameters.project_name, continue_last_task=True  # FIXME?
    )

    clearml_task.connect_configuration(config, name='HyperParameters')

    return clearml_task
