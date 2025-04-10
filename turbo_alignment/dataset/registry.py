from turbo_alignment.common.registry import Registrable
from turbo_alignment.settings.datasets import DatasetType


class DatasetRegistry(Registrable):
    ...


@DatasetRegistry.register(DatasetType.PAIR_PREFERENCES)
class PairPreferenceDatasetTypeRegistry(Registrable):
    ...


@DatasetRegistry.register(DatasetType.CHAT)
class ChatDatasetTypeRegistry(Registrable):
    ...


@DatasetRegistry.register(DatasetType.CLASSIFICATION)
class ClassificationDatasetTypeRegistry(Registrable):
    ...


@DatasetRegistry.register(DatasetType.SAMPLING)
class SamplingRMDatasetTypeRegistry(Registrable):
    ...
