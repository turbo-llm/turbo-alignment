from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset
from transformers import EvalPrediction

from turbo_alignment.settings.pipelines.train.classification import (
    ClassificationLossSettings,
)
from turbo_alignment.trainers.custom_loss import CustomLossTrainer


def compute_clf_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    pred_scores, labels = eval_pred
    predictions = np.argmax(pred_scores, axis=1)
    accuracy = accuracy_score(labels, predictions)
    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    f_score = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    roc_auc = roc_auc_score(labels, pred_scores[:, 1], average='macro', multi_class='ovo')
    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1-score': f_score,
        'recall': recall,
        'precision': precision,
        'roc_auc': roc_auc,
    }
    return metrics


def classification_loss(
    logits: torch.Tensor, labels: torch.LongTensor, loss_settings: ClassificationLossSettings
) -> torch.Tensor:
    if loss_settings.alpha is None:
        alpha = torch.ones((logits.size(-1),), device=logits.device, dtype=logits.dtype)
    else:
        alpha = torch.tensor(loss_settings.alpha, device=logits.device, dtype=logits.dtype)

    ce_loss = F.cross_entropy(logits, labels, weight=alpha, reduction='none')

    p_t = torch.exp(-ce_loss)

    focal_loss = ((1 - p_t) ** loss_settings.gamma) * ce_loss

    return focal_loss.mean()


def auto_class_weights(dataset: Dataset) -> list[float]:
    labels = [dataset[i]['labels'] for i in range(len(dataset))]  # type: ignore[arg-type]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=np.array(labels))
    return class_weights.tolist()


class ClassificationTrainer(CustomLossTrainer):
    def __init__(self, loss_settings: ClassificationLossSettings, **kwargs):
        super().__init__(
            custom_loss=partial(classification_loss, loss_settings=loss_settings),
            compute_metrics=compute_clf_metrics,
            **kwargs,
        )
