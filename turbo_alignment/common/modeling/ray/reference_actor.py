import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turbo_alignment.common.tf.loaders.model.model import disable_dropout_in_model
from turbo_alignment.dist_utils.ray.distributed_torch_ray_actor import (
    DistributedTorchRayActor,
)


@ray.remote(num_gpus=1)
class ReferenceModel(DistributedTorchRayActor):
    def __init__(self, world_size, rank, local_rank, master_addr, master_port):
        super().__init__(world_size, rank, local_rank, master_addr, master_port)
        self.node_id = ray.get_runtime_context().get_node_id()
        self.local_rank = ray.get_gpu_ids()

    def init_model_from_pretrained(self, pretrain):
        self._setup_distributed()
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrain,
            device_map='cuda',
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',  # FIXME hardcoding all this
        )

        for _, param in self.model.named_parameters():
            param.requires_grad = False

        disable_dropout_in_model(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True)
        print(f'Reference model initialized on Node {self.node_id}, Local Rank {self.local_rank}')
        print('GPU IDs: {}'.format(ray.get_runtime_context().get_accelerator_ids()['GPU']))

    def prepare_reference_model(self, accelerator, is_deepspeed_enabled):
        if is_deepspeed_enabled:
            import deepspeed

            self.model, *_ = deepspeed.initialize(
                model=self.model,
                # FIXME
                config={
                    'zero_optimization': {'stage': 0},
                    'optimizer': {'type': None},
                    'fp16': {'enabled': False},
                    'bf16': {'enabled': True},
                    'train_micro_batch_size_per_gpu': 1,
                },
            )
            self.model.eval()
        else:
            self.model = accelerator.prepare_model(self.model, evaluation_mode=True)

    # def tokenize(self, text: str):
    #     return self.tokenizer(text, return_tensors='pt')

    # def generate(self, text: str):
    #     tokenized_input = self.tokenize(text).to('cuda')
    #     return self.model(**tokenized_input)

    # def eval(self):
    #     return self.model.eval()

    # @torch.no_grad
    # def forward(self, x):
    #     self.model.eval()
    #     x = {k: v.cuda() for k, v in x.items()}
    #     return self.model(**x)

    # FIXME
    @torch.no_grad  # type: ignore[call-arg]
    def reference_forward(self, x):
        # torch.cuda.empty_cache() #FIXME
        self.model.eval()
        x = {k: v.cuda() for k, v in x.items()}

        logits = self.model(**x).logits[:, :-1]

        return logits
