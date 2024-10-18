import torch
import torch.nn.functional as F
from transformers import DataCollatorForTokenClassification


class DataCollatorWithModalityInputs(DataCollatorForTokenClassification):
    def torch_call(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        if 'modality_inputs' in features[0].keys():
            # print([feature['modality_inputs'] for feature in features])
            modality_inputs = torch.stack(
                [torch.stack(feature['modality_inputs']) for feature in features]
            ).contiguous()
        else:
            modality_inputs = [None for _ in features]

        if 'messages' in features[0].keys():
            message_inputs = [feature['messages'] for feature in features]
        else:
            message_inputs = [None] * len(features)

        if 'messages' in features[0].keys():
            modality_input_names = (label_name, 'modality_inputs', 'modality_tokens_mask', 'messages')
        else:
            modality_input_names = (label_name, 'modality_inputs', 'modality_tokens_mask')
        tokenizer_features = [
            {k: v for k, v in feature.items() if k not in modality_input_names} for feature in features
        ]

        batch = self.tokenizer.pad(
            tokenizer_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )

        sequence_length = batch['input_ids'].shape[1]
        padding_side = self.tokenizer.padding_side

        assert padding_side == 'right'

        batch['modality_inputs'] = modality_inputs

        if 'messages' in features[0].keys():
            batch['messages'] = message_inputs

        batch['modality_tokens_mask'] = torch.stack(
            [
                F.pad(
                    feature['modality_tokens_mask'],
                    pad=(0, sequence_length - len(feature['modality_tokens_mask'])),
                    mode='constant',
                    value=0,
                )
                for feature in features
            ]
        ).contiguous()

        if labels is None:
            return batch

        batch[label_name] = [
            label.tolist() + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
        ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64).contiguous()

        return batch
