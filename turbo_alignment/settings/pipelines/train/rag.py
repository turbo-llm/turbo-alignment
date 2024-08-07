from turbo_alignment.settings.cherry_pick import ChatCherryPickSettings
from turbo_alignment.settings.datasets.chat import ChatMultiDatasetSettings
from turbo_alignment.settings.pipelines.train.base import BaseTrainExperimentSettings
from turbo_alignment.settings.rag_model import RAGPreTrainedModelSettings


class RAGTrainExperimentSettings(BaseTrainExperimentSettings):
    model_settings: RAGPreTrainedModelSettings  # type: ignore[assignment]
    train_dataset_settings: ChatMultiDatasetSettings
    val_dataset_settings: ChatMultiDatasetSettings

    cherry_pick_settings: ChatCherryPickSettings
