from typing import Any

from turbo_alignment.settings.logging.clearml import ClearMLSettings

# from clearml import Task



def create_clearml_task(parameters: ClearMLSettings, config: dict[str, Any] | None = None):
    # clearml_task = Task.init(
    #     task_name=parameters.task_name,
    #     project_name=parameters.project_name,
    #     continue_last_task=True,
    #     output_uri=False,
    # )

    # clearml_task.connect_configuration(config, name='HyperParameters')

    # return clearml_task
    return None
