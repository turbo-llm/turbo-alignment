import argparse
import pathlib

from turbo_alignment.pipelines.train.dpo import TrainDPOStrategy
from turbo_alignment.pipelines.train.sft import TrainSFTStrategy
from turbo_alignment.settings.pipelines.train.dpo import DPOTrainExperimentSettings
from turbo_alignment.settings.pipelines.train.sft import SftTrainExperimentSettings

TASK_TYPE_TO_STRATEGY = {
    'dpo': (TrainDPOStrategy, DPOTrainExperimentSettings),
    'sft': (TrainSFTStrategy, SftTrainExperimentSettings),
}


def run_pipeline(
    settings: DPOTrainExperimentSettings | SftTrainExperimentSettings,
    pipeline_cls: TrainDPOStrategy | TrainSFTStrategy,
):
    print(settings)
    pipeline_cls().run(settings)


def run(task_type: str, settings_path: pathlib.Path, make_model_vanilla: bool):
    strategy_cls, settings_cls = TASK_TYPE_TO_STRATEGY[task_type]

    experiment_settings = settings_cls.parse_file(settings_path)
    if make_model_vanilla:
        vanilla_settings = experiment_settings.copy(deep=True)
        vanilla_settings.trainer_settings.sequence_parallel = 1
        vanilla_settings.model_settings.model_kwargs["attn_implementation"] = "eager"
        vanilla_settings.model_settings.model_type = 'causal'

        experiment_settings = vanilla_settings

    experiment_settings.trainer_settings.load_best_model_at_end = False

    run_pipeline(experiment_settings, strategy_cls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-type', required=True, choices=list(TASK_TYPE_TO_STRATEGY.keys()))
    parser.add_argument('--settings-path', required=True, type=pathlib.Path)
    parser.add_argument('--make-model-vanilla', action='store_true')
    args = parser.parse_args()

    run(args.task_type, args.settings_path, args.make_model_vanilla)


if __name__ == '__main__':
    main()
