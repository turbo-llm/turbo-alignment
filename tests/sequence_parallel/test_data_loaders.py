import pytest
import torch
from transformers import Trainer, TrainingArguments

from turbo_alignment.modeling import parallel_states
from turbo_alignment.sequence_parallel.patch_accelerate import patch_acclerator

from tests.sequence_parallel.consts import DEEPSPEED_CONFIG
from tests.sequence_parallel.dataset import SimpleDataset
from tests.sequence_parallel.launcher import app, launch_with_name
from tests.sequence_parallel.marks import has_two_gpus


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.w * x


@app.command(name='with-seq-p')
def run_with_seq_p():
    dataset = SimpleDataset([{'x': torch.tensor([i])} for i in range(2)])
    with patch_acclerator():
        args = TrainingArguments(
            output_dir='.',
            do_train=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            deepspeed=DEEPSPEED_CONFIG,
        )

        parallel_states.initialize_model_parallel(sequence_parallel_size=2)

        model = SimpleModel()

        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            args=args,
        )

        for i, batch in enumerate(trainer.get_train_dataloader()):
            assert batch['x'].item() == i, batch


@app.command('without-seq-p')
def run_without_seq_p():
    dataset = SimpleDataset([{'x': torch.tensor([i])} for i in range(2)])
    with patch_acclerator():
        args = TrainingArguments(
            output_dir='.',
            do_train=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            deepspeed=DEEPSPEED_CONFIG,
        )

        parallel_states.initialize_model_parallel(sequence_parallel_size=1)

        model = SimpleModel()

        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            args=args,
        )

        for batch in trainer.get_train_dataloader():
            assert batch['x'].item() == torch.distributed.get_rank(), batch


def main():
    return app()


@pytest.mark.gpu
@pytest.mark.skipif(not has_two_gpus(), reason='at least two gpus are required')
@pytest.mark.parametrize(
    'cmd_name',
    [
        pytest.param('with-seq-p', id='with-seq-p'),
        pytest.param('without-seq-p', id='without-seq-p'),
    ],
)
def test_data_loader(cmd_name):
    return launch_with_name(__file__, cmd_name, 2)


if __name__ == '__main__':
    main()
