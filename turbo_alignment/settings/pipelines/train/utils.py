from enum import Enum
from typing import Literal

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


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


class APOZeroLossSettings(DPOLossSettings):
    loss_type: Literal[DPOLossesType.APO_ZERO]


class APODownLossSettings(DPOLossSettings):
    loss_type: Literal[DPOLossesType.APO_DOWN]
