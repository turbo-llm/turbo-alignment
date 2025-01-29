import ray
from turbo_alignment.trainers.online.ray.distributed_torch_ray_actor import DistributedTorchRayActor
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import gc


def sum_all_parameter_values(model):
    # if isinstance(model, deepspeed.DeepSpeedEngine):
        # model = model.module
    total_sum = sum(p.sum().item() for p in model.parameters())
    return total_sum


@ray.remote(num_gpus=1)
class ReferenceModel(DistributedTorchRayActor):
    def __init__(self, world_size, rank, local_rank, master_addr, master_port):
        super().__init__(world_size, rank, local_rank, master_addr, master_port)
        self.node_id = ray.get_runtime_context().get_node_id()
        self.local_rank = ray.get_gpu_ids()
    
    def init_model_from_pretrained(self, pretrain):
        self._setup_distributed()
        self.model = AutoModelForCausalLM.from_pretrained(pretrain, device_map='cuda', torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2') #attn_implementation='flash_attention_2'
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True)
        print('IN REFERENCE NODE REF MODEL WEIGHT , ', sum_all_parameter_values(self.model))
        # print(f"Reference model initialized on Node {self.node_id}, Local Rank {self.local_rank}")
        # print("GPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["GPU"]))

    def tokenize(self, text: str):
        return self.tokenizer(text, return_tensors='pt')
    
    def generate(self, text: str):
        tokenized_input = self.tokenize(text).to('cuda')
        return self.model(**tokenized_input)
    
    def eval(self):
        return self.model.eval()
    
    @torch.no_grad
    def forward(self, x):
        self.model.eval()
        print('IN REFERENCE NODE REF MODEL WEIGHT HERE IM FORWARD HERE, ', sum_all_parameter_values(self.model))
        x = {k: v.cuda() for k, v in x.items()}
        return self.model(**x)
    
    @torch.no_grad
    def reference_forward(self, x):
        torch.cuda.empty_cache()
        self.model.eval()
        x = {k: v.cuda() for k, v in x.items()}

        # print(f"{x.keys()}")
        logits = self.model(**x).logits

        return logits

        # logits /= temperature

        # '''
        # Memory Efficient implementation of log_softmax using in_place operation
        # equivalent to:
        # logits = F.log_softmax(logits, dim=-1)
        # '''
        # torch.exp(logits, out=logits)
        # summed = torch.sum(logits, dim=-1, keepdim=True)
        # logits /= summed
        # torch.log(logits, out=logits)

        # logprob = torch.gather(logits, 2, x['input_ids'][:, 1:].unsqueeze(-1)).squeeze(-1)
        # logprob[~loss_mask[:, 1:].to(torch.bool)] = 0
        # out = logprob.sum(-1)

        # return out