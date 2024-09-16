import abc

import torch
from torch.nn.modules import Module
from transformers import PreTrainedModel, AutoTokenizer
from typing import Literal
import deepspeed
from abc import ABC
import ray


from allenai_common import Registrable

from turbo_alignment.settings.online import LLMActorType


class LLMActor(ABC, Registrable):

    def __init__(
        self,
        tokenizer: AutoTokenizer,         
        max_tokens_count: int,
        stop_token_id: int,
        temperature: float,
    ) -> None:
        self.tokenizer = tokenizer,
        self.max_tokens_count: int = max_tokens_count
        self.stop_token_id: int = stop_token_id
        self.temperature: float = temperature

    @abc.abstractmethod
    def generate_responses(
        self, 
        model: torch.nn.Module | PreTrainedModel,
        queries: torch.Tensor,
        attention_mask: torch.Tensor,
        train_eval: Literal["train", "eval"],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        query_responses,
        attention_mask,
        response_tokens_mask,
        position_ids,
        rewards,
        meta
        """
        
        raise NotImplementedError


@LLMActor.register(LLMActorType.LOCAL_TRANSFORMERS)
class LocalTransformersLLMActor(LLMActor):

    def generate_responses(
        self, 
        model: torch.nn.Module | PreTrainedModel, 
        queries: torch.Tensor, 
        attention_mask: torch.Tensor, 
        train_eval: Literal['train'] | Literal['eval']
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = model.generate(inputs=queries, attention_mask=attention_mask)

        ...


@LLMActor.register(LLMActorType.DISTRIBUTED_VLLM)
class DistributedVLLMActor(LLMActor):

    def __init__(self):
        self.vllm_engines = ...

    def _broadcast_to_vllm(self):
        # avoid OOM
        torch.cuda.empty_cache()
        model = self.actor.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # Fire all vllm engines for broadcast
            if torch.distributed.get_rank() == 0:
                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight.remote(name, dtype=param.dtype, shape=shape, empty_cache=count == num_params)
                    for engine in self.vllm_engines
                ]

            # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
            with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                if torch.distributed.get_rank() == 0:
                    torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
                    ray.get(refs)

    def generate_responses(
        self, 
        model: Module | PreTrainedModel, 
        queries: torch.Tensor, 
        attention_mask: torch.Tensor, 
        train_eval: Literal['train'] | Literal['eval']
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

    def _generate_vllm(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank()
        llm = self.vllm_engines[rank % len(self.vllm_engines)]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
        )

        # TODO: can't pass `max_length` to vLLM's tokenizer for input truncation, remove this once it is supported.
        input_ids = self.tokenize_fn(prompts, self.prompt_max_len, device="cpu")["input_ids"]
        assert self.tokenizer.padding_side == "left", f"tokenizer padding_size should be left"
        pad_indices = (input_ids != self.tokenizer.pad_token_id).to(dtype=torch.int).argmax(dim=-1)
        prompt_token_ids = []
        for i, pad_index in enumerate(pad_indices.numpy()):
            prompt_token_ids.append(input_ids[i][pad_index:].tolist())
        outputs = ray.get(llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))

        # NOTE: concat all outputs to following format:
        #
        # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
        # | token token token token token | token token [EOS] [PAD] |
        # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
        # |<---------- prompt ----------->|<-------- answer ------->|
        max_input_len, max_output_len = 0, 0
        for output in outputs:
            max_input_len = max(max_input_len, len(output.prompt_token_ids))
            max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        sequences = []
        for output in outputs:
            # left padding input
            input_len = len(output.prompt_token_ids)
            input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

            # right padding output
            output_len = len(output.outputs[0].token_ids)
            output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

            if output_ids[output_len - 1] != eos_token_id:
                output_ids[min(output_len, len(output_ids) - 1)] = eos_token_id

            # concat input and output
            sequences.append(input_ids + output_ids)

        sequences = torch.tensor(sequences)
        sequences, attention_mask, action_mask = self.actor.process_sequences(
            sequences, max_input_len, eos_token_id, pad_token_id
        )
        return sequences.to("cuda"), attention_mask.to("cuda"), action_mask.to("cuda")