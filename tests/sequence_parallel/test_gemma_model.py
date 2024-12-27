import os

import deepspeed
import deepspeed.comm as dist
import pytest
import torch
import torch.distributed
import turbo_alignment.modeling.parallel_states as parallel_states
from transformers import AutoTokenizer, Trainer
from transformers.data.data_collator import default_data_collator

from turbo_alignment.dist_utils.gather_and_split import all_gather_variable
from turbo_alignment.modeling.gemma2.patch import patch_gemma_attn_dict
from turbo_alignment.modeling.gemma import (
    Gemma2ForCausalLM,
    Gemma2ForCausalLMWithMPU,
)
from turbo_alignment.sequence_parallel.collator import (
    DataCollatorForSequenceParallism,
    pad_for_sequence_parallel,
)
from turbo_alignment.sequence_parallel.patch_accelerate import patch_acclerator
from turbo_alignment.trainers.base_args import TrainingArgumentsWithSeqP

from tests.sequence_parallel.consts import DEEPSPEED_CONFIG, MODEL_PATH
from tests.sequence_parallel.dataset import SimpleDataset
from tests.sequence_parallel.launcher import app, launch_with_name
from tests.sequence_parallel.marks import has_gemma_model, has_two_gpus


import functools


def run_in_order(group=None):
    def inner(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            rank = dist.get_rank(group)
            for i in range(dist.get_world_size(group)):
                if i == rank:
                    res = f(*args, **kwargs)

                dist.barrier(group)

            return res

        return wrapped

    return inner


@app.command(name='gemma-model')
def gemma_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        pytest.skip(f'directory {model_path} not found')
        return

    if not os.path.isdir(model_path):
        raise ValueError(f'Model path {model_path} is not a directory')

    patch_gemma_attn_dict()

    dist.init_distributed()

    parallel_states.initialize_model_parallel(sequence_parallel_size=2)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    current_device = f'cuda:{dist.get_rank()}'

    vanilla_model = Gemma2ForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    ).to(current_device)

    model = Gemma2ForCausalLMWithMPU.from_pretrained(
        model_path,
        attn_implementation="eager_ulysses",
        torch_dtype=torch.float32,
    ).to(current_device)

    config = DEEPSPEED_CONFIG
    config['bf16']['enabled'] = False

    model, *_ = deepspeed.initialize(model=model, mpu=parallel_states, config=config)
    model.train()

    tokenized = tokenizer('Мама мыла раму. ' * 5, return_tensors='pt').to(model.device)
    input_ids = pad_for_sequence_parallel(
        tokenized['input_ids'], parallel_states.get_sequence_parallel_world_size(), 0
    )
    attention_mask = pad_for_sequence_parallel(
        tokenized['attention_mask'], parallel_states.get_sequence_parallel_world_size(), False
    )

    seq_len = input_ids.size(1)
    chunk_size = seq_len // parallel_states.get_sequence_parallel_world_size()

    cache_position = torch.arange(0, input_ids.shape[1], device=input_ids.device)
    position_ids = cache_position.unsqueeze(0)

    seq_len = input_ids.size(1)
    chunk_size = seq_len // parallel_states.get_sequence_parallel_world_size()
    start = chunk_size * parallel_states.get_sequence_parallel_rank()
    end = chunk_size * (parallel_states.get_sequence_parallel_rank() + 1)
    result = model(
        input_ids[:, start:end],
        attention_mask,
        position_ids=position_ids,
    ).logits

    result.mean().backward()

    vanilla_result = vanilla_model(
        input_ids,
        attention_mask,
        position_ids=position_ids,
        use_cache=False,
    ).logits

    vanilla_result.mean().backward()

    torch.testing.assert_close(result, vanilla_result[:, start:end], atol=0.01, rtol=0.01)

    if dist.get_rank() == 0 or True:
        print('####BEGIN')
        for name, param in model.module.lm_head.named_parameters():
            print(name, param)
        print('#####END')

        for param in model.module.lm_head.parameters():
            print(param)

        print('#####END 2')

        for name, param in vanilla_model.named_parameters():
            print(name)
            if name.startswith('module'):
                name = name[len('module.') :]

            v_param = vanilla_model.get_parameter(name)
            assert param.requires_grad == v_param.requires_grad
            if param.requires_grad:
                print('Do for', name)

                torch.testing.assert_close(
                    param.grad,
                    v_param.grad,
                    atol=0.5,
                    rtol=0.2,
                    msg=lambda msg: '\n'.join([name, msg]),
                )
            else:
                print('Skip', name)


@app.command(name='test-dataloader')
def _test_dataloader(model_path: str = MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    texts = [
        'Мама мыла раму',
    ]
    dataset = SimpleDataset([tokenizer(text) for text in texts])

    patch_gemma_attn_dict()

    with patch_acclerator():
        model = Gemma2ForCausalLMWithMPU.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2_ulysses",
            torch_dtype=torch.bfloat16,
        )

        vanilla_model = Gemma2ForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )

        args = TrainingArgumentsWithSeqP(
            output_dir='/mnt/models/p.geyn/',
            do_train=True,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            deepspeed=DEEPSPEED_CONFIG,
            sequence_parallel=2,
            gradient_checkpointing=False,
            report_to='none',
            save_total_limit=0,
        )

        model = model.to(args.device)

        vanilla_model = vanilla_model.to(args.device)

        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            args=args,
            data_collator=DataCollatorForSequenceParallism(
                base_collate_fn=default_data_collator,
                seq_p_rank=parallel_states.get_sequence_parallel_rank(),
                seq_p_world_size=parallel_states.get_sequence_parallel_world_size(),
            ),
        )

        print('Alive')
        for batch in trainer.get_train_dataloader():
            print(batch)
            generated = model.generate(**batch, max_new_tokens=2, use_cache=False)
            # run_in_order()(print)('rank:', dist.get_rank(), 'result:', generated)
            all_generated = torch.cat(
                all_gather_variable(generated, parallel_states.get_sequence_parallel_group()),
                dim=-1,
            )

            run_in_order()(print)('rank:', dist.get_rank(), all_generated)
            all_input_ids = torch.cat(
                all_gather_variable(batch['input_ids'], parallel_states.get_sequence_parallel_group()),
                dim=-1,
            )
            vanilla_result = vanilla_model.generate(all_input_ids, max_new_tokens=2)
            run_in_order()(print)('rank:', dist.get_rank(), vanilla_result)
            assert torch.equal(all_generated, vanilla_result), (all_generated, vanilla_result)




@pytest.mark.skipif(not has_two_gpus(), reason='At least two gpu are required')
@pytest.mark.skipif(not has_gemma_model(), reason='Gemma model not found')
def test_dataloader():
    return launch_with_name('test-dataloader', 2)


@pytest.mark.skipif(not has_two_gpus(), reason='At least two gpu are required')
@pytest.mark.skipif(not has_gemma_model(), reason='Gemma model not found')
def test_gemma_model():
    return launch_with_name('gemma-model', 2)


if __name__ == '__main__':
    def set_prctl():
        import prctl
        prctl.set_ptracer(prctl.SET_PTRACER_ANY)

    import os
    set_prctl()
    os.register_at_fork(after_in_child=set_prctl)
    app()
