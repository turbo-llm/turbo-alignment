from turbo_alignment.settings.pipelines.train.dpo import (
    DPOTrainerSettings,
    DPOTrainExperimentSettings,
)


class LDDPOTrainerSettings(DPOTrainerSettings):
    alpha: float = 1.0


class LDDPOTrainExperimentSettings(DPOTrainExperimentSettings):
    trainer_settings: LDDPOTrainerSettings
