import ray
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from turbo_alignment.trainers.online.ray.distributed_torch_ray_actor import (
    DistributedTorchRayActor,
)


@ray.remote(num_gpus=1)
class RewardModel(DistributedTorchRayActor):
    def __init__(self, world_size, rank, local_rank, master_addr, master_port):
        super().__init__(world_size, rank, local_rank, master_addr, master_port)
        self.node_id = ray.get_runtime_context().get_node_id()
        self.local_rank = ray.get_gpu_ids()

    def init_model_from_pretrained(self, rm_model):
        self._setup_distributed()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            rm_model,
            num_labels=1,  ##FIXME hardcoding all this
            device_map='cuda',
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            rm_model,
            trust_remote_code=True,  # FIXME!!!
        )

        self.model.config.pad_token_id = 151643  # FIXME
        self.tokenizer.pad_token = '<|endoftext|>'  # FIXME

        self.model.eval()

        print(f'Reward model initialized on Node {self.node_id}, Local Rank {self.local_rank}')
        print('GPU IDs: {}'.format(ray.get_runtime_context().get_accelerator_ids()['GPU']))

    def tokenize(self, text: str):
        return self.tokenizer(text, return_tensors='pt')

    def generate(self, text: str):
        tokenized_input = self.tokenize(text).to('cuda')
        return self.model(**tokenized_input)

    def eval(self):
        return self.model.eval()

    @torch.no_grad
    def forward(self, x):
        torch.cuda.empty_cache()
        x = {k: v.cuda() for k, v in x.items()}

        max_len = x['input_ids'].size(0)

        # FIXME!!!!
        eoses = (
            self.tokenizer.encode('\n<|im_end|>', return_tensors='pt', add_special_tokens=False)
            .repeat(max_len, 1)
            .cuda()
        )
        eoses_attn_mask = torch.ones(eoses.shape, device=x['input_ids'].device)

        x['input_ids'] = torch.concat((x['input_ids'], eoses), dim=1)
        x['attention_mask'] = torch.concat((x['attention_mask'], eoses_attn_mask), dim=1)

        position_ids = (x['attention_mask'].cumsum(-1) - 1).clamp(min=0)
        position_ids.masked_fill_(x['attention_mask'].to(torch.bool) == 0, 0).cuda()

        return self.model(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            position_ids=position_ids,
        ).logits
