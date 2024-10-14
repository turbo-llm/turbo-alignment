from enum import Enum
from typing import Literal

from pydantic import Field

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.cherry_pick import ChatCherryPickSettings
from turbo_alignment.settings.datasets.pair_preference import (
    PairPreferenceMultiDatasetSettings,
)
from turbo_alignment.settings.pipelines.train.base import BaseTrainExperimentSettings
from turbo_alignment.settings.tf.trainer import TrainerSettings


class DPOLossesType(str, Enum):
    SIGMOID = 'sigmoid'
    SIGMOID_WITH_MARGIN = 'sigmoid_with_margin'
    HINGE = 'hinge'
    IPO = 'ipo'
    KTO = 'kto'
    SLIC_HF = 'slic_hf'
    CPO = 'cpo'
    ORPO = 'orpo'
    SIMPO = 'simpo'
    APO_ZERO = 'apo_zero'
    APO_DOWN = 'apo_down'
    ASFT = 'asft'


class DPOLossSettings(ExtraFieldsNotAllowedBaseModel):
    loss_type: DPOLossesType
    beta: float = 0.1


class KTOLossSettings(DPOLossSettings):
    loss_type: Literal[DPOLossesType.KTO]


class SigmoidLossSettings(DPOLossSettings):
    loss_type: Literal[DPOLossesType.SIGMOID]
    label_smoothing: float = 0


class SigmoidLossWithMarginSettings(DPOLossSettings):
    loss_type: Literal[DPOLossesType.SIGMOID_WITH_MARGIN]


class HingeLossSettings(DPOLossSettings):
    loss_type: Literal[DPOLossesType.HINGE]
    norm: bool = True


class IPOLossSettings(DPOLossSettings):
    loss_type: Literal[DPOLossesType.IPO]


class CPOLossSettings(DPOLossSettings):
    loss_type: Literal[DPOLossesType.CPO]
    norm: bool = True


class SlicHfLossSettings(DPOLossSettings):
    loss_type: Literal[DPOLossesType.SLIC_HF]
    delta: float = 1.0
    lam: float = 0.1
    norm: bool = False


class SimPOLossSettings(DPOLossSettings):
    loss_type: Literal[DPOLossesType.SIMPO]
    gamma: float = 0.1


class ORPOLossSettings(DPOLossSettings):
    loss_type: Literal[DPOLossesType.ORPO]


class ASFTLossSettings(DPOLossSettings):
    loss_type: Literal[DPOLossesType.ASFT]


class APOZeroLossSettings(DPOLossSettings):
    loss_type: Literal[DPOLossesType.APO_ZERO]


class APODownLossSettings(DPOLossSettings):
    loss_type: Literal[DPOLossesType.APO_DOWN]


class SyncRefModelSettings(ExtraFieldsNotAllowedBaseModel):
    sync_ref_model: bool = False
    alpha: float = 1.0
    sync_steps: int = 1


class DPOTrainerSettings(TrainerSettings):
    loss_settings: (
        SigmoidLossSettings
        | HingeLossSettings
        | IPOLossSettings
        | KTOLossSettings
        | CPOLossSettings
        | ORPOLossSettings
        | ASFTLossSettings
        | SimPOLossSettings
        | SlicHfLossSettings
        | SigmoidLossWithMarginSettings
        | APOZeroLossSettings
        | APODownLossSettings
    )
    sync_ref_settings: SyncRefModelSettings
    use_ref_model: bool = True
    use_sft_model: bool = False
    average_log_prob: bool = Field(default=False, description='Normalize log probability by length or not')


class DPOTrainExperimentSettings(BaseTrainExperimentSettings):
    train_dataset_settings: PairPreferenceMultiDatasetSettings
    val_dataset_settings: PairPreferenceMultiDatasetSettings

    cherry_pick_settings: ChatCherryPickSettings

    trainer_settings: DPOTrainerSettings
