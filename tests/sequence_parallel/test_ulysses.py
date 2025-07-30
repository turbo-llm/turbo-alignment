# pylint: disable=protected-access

import copy

import pytest
import torch
from transformers import TrainingArguments
from transformers.trainer import Trainer
from transformers.models.gemma2 import Gemma2Config, Gemma2Model
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2RotaryEmbedding,
    Gemma2Attention as VanillaGemma2Attention,
)
from turbo_alignment.modeling import parallel_states
from turbo_alignment.sequence_parallel.patch_accelerate import patch_acclerator
from turbo_alignment.modeling.gemma import Gemma2Attention as UlyssesGemma2Attention
from turbo_alignment.common import set_random_seed

from tests.sequence_parallel.consts import DEEPSPEED_CONFIG
from tests.sequence_parallel.dataset import SimpleDataset
from tests.sequence_parallel.launcher import app, launch_with_name
from tests.sequence_parallel.marks import has_two_gpus


CONFIG = Gemma2Config()


def fix_attention_mask(
    attention_mask,
    input_tensor,
    cache_position: torch.Tensor,
):
    dtype, device = input_tensor.dtype, input_tensor.device
    sequence_length = input_tensor.shape[1]
    sequence_length = attention_mask.shape[-1]
    target_length = attention_mask.shape[-1] if attention_mask is not None else input_tensor.shape[1]

    # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
    # pylint: disable=protected-access
    causal_mask = Gemma2Model._prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask,
        sequence_length=sequence_length,
        target_length=target_length,
        dtype=dtype,
        device=device,
        cache_position=cache_position,
        batch_size=input_tensor.shape[0],
    )
    return causal_mask


def run_with_seq_p(config=CONFIG, seq_len=10, num_items: int = 6, seed: int = 42, dtype=torch.float32):
    set_random_seed(seed)

    dataset = SimpleDataset(
        [
            {
                'hidden_states': torch.randn((seq_len, config.hidden_size), dtype=dtype),
                'attention_mask': torch.tensor([True] * num_items + [False] * (seq_len - num_items)),
            }
            for _ in range(2)
        ]
    )

    with patch_acclerator():
        args = TrainingArguments(
            output_dir='.',
            do_train=True,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            deepspeed=DEEPSPEED_CONFIG,
        )

        parallel_states.initialize_model_parallel(sequence_parallel_size=2)

        set_random_seed(seed)
        model = UlyssesGemma2Attention(config, 0).to(dtype)

        set_random_seed(seed)
        vanilla = VanillaGemma2Attention(config, layer_idx=0).to(dtype)

        torch.testing.assert_close(model.k_proj.weight, vanilla.k_proj.weight)
        torch.testing.assert_close(model.v_proj.weight, vanilla.v_proj.weight)
        torch.testing.assert_close(model.q_proj.weight, vanilla.q_proj.weight)
        torch.testing.assert_close(model.o_proj.weight, vanilla.o_proj.weight)

        model = model.to('cuda')
        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            args=args,
        )

        for batch in trainer.get_train_dataloader():
            q = batch['hidden_states']
            rot_emb = Gemma2RotaryEmbedding(config, device=q.device)
            cache_position = torch.arange(0, q.shape[1], device=q.device)
            position_ids = cache_position.unsqueeze(0)

            start = (seq_len // 2) * parallel_states.get_sequence_parallel_rank()
            end = (seq_len // 2) * (parallel_states.get_sequence_parallel_rank() + 1)

            attention_mask = batch['attention_mask']

            if config._attn_implementation.startswith('eager'):
                attention_mask = fix_attention_mask(attention_mask, q, cache_position)

            output = model(
                q[:, start:end],
                position_embeddings=rot_emb(q, position_ids[:, start:end]),
                attention_mask=attention_mask,
            )[0]

            loss = torch.nn.functional.mse_loss(output, torch.zeros_like(output))
            loss.backward()

            print('Compute vanilla')

            vanilla = vanilla.to(q.device)
            vanilla_output = vanilla(
                q,
                position_embeddings=rot_emb(q, position_ids),
                attention_mask=attention_mask,
            )[0]
            print(f'{vanilla_output.size()=}')
            vanilla_loss = torch.nn.functional.mse_loss(vanilla_output, torch.zeros_like(vanilla_output))
            vanilla_loss.backward()

            vanilla_subset = vanilla_output[:, start : min(end, num_items)]
            output_subset = output[:, : min(end, num_items - start)]

            assert vanilla_subset.size() == output_subset.size(), (vanilla_subset.size(), output_subset.size())
            torch.testing.assert_close(output_subset, vanilla_subset, atol=0.3, rtol=2)

            assert model.k_proj.weight.grad is not None
            torch.testing.assert_close(model.k_proj.weight.grad, vanilla.k_proj.weight.grad, atol=0.3, rtol=0.2)
            torch.testing.assert_close(model.v_proj.weight.grad, vanilla.v_proj.weight.grad, atol=0.3, rtol=0.2)
            torch.testing.assert_close(model.q_proj.weight.grad, vanilla.q_proj.weight.grad, atol=0.3, rtol=0.2)

            # torch.distributed.barrier()


@app.command('ulysses-with-flash')
def with_flash():
    config = copy.deepcopy(CONFIG)
    config._attn_implementation = 'flash_attention_2'
    run_with_seq_p(config, dtype=torch.bfloat16)


@app.command('ulysses-without-flash')
def without_flash():
    run_with_seq_p()


@pytest.mark.skipif(not has_two_gpus(), reason='at least two gpus are required')
@pytest.mark.parametrize(
    'cmd_name',
    [
        pytest.param('ulysses-with-flash', id='flash'),
        pytest.param('ulysses-without-flash', id='eager'),
    ],
)
def test_ulysses_attention(cmd_name):
    return launch_with_name(__file__, cmd_name, 2)


def main():
    app()


if __name__ == '__main__':
    main()
