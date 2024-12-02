import ray
from turbo_alignment.trainers.online.ray.distributed_torch_ray_actor import DistributedTorchRayActor
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import gc


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


@ray.remote(num_gpus=1)
class ReferenceModel(DistributedTorchRayActor):
    def __init__(self, world_size, rank, local_rank, master_addr, master_port):
        super().__init__(world_size, rank, local_rank, master_addr, master_port)
        self.node_id = ray.get_runtime_context().get_node_id()
        self.local_rank = ray.get_gpu_ids()
    
    def init_model_from_pretrained(self, pretrain):
        self._setup_distributed()
        self.model = AutoModelForCausalLM.from_pretrained(pretrain, device_map='cuda', torch_dtype=torch.bfloat16) #attn_implementation='flash_attention_2'

        disable_dropout_in_model(self.model)

        hash = 0
        for p in self.model.parameters():
            hash += p.data.sum().item()
        print("RAY REFERENCE MODEL HASH: ", hash)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True)
        self.model.resize_token_embeddings(len(self.tokenizer))
        print(f"Reference model initialized on Node {self.node_id}, Local Rank {self.local_rank}")
        print("GPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["GPU"]))

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
        x = {k: v.cuda() for k, v in x.items()}
        return self.model(**x)
    
    @torch.no_grad
    def reference_forward(self, x, temperature, loss_mask):
        torch.cuda.empty_cache()
        x = {k: v.cuda() for k, v in x.items()}

        print(f"{x.keys()}")
        # logits = self.model(**x).logits[:, :-1] # 35GB
        # logits /= temperature

        #logits = F.log_softmax(logits, dim=-1) # 35GB
        '''
        Memory Efficient implementation of log_softmax using in_place operation
        '''
        # torch.exp(logits, out=logits)
        # summed = torch.sum(logits, dim=-1, keepdim=True)
        # logits /= summed
        # torch.log(logits, out=logits)

        raw_logits = self.model(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            position_ids=x['position_ids'],
            use_cache=False,
        ).logits[:, :-1] / temperature

        with autocast(dtype=torch.float32):
            logits = F.log_softmax(
                raw_logits,
                dim=-1
            )

        print(f"LOGITS FROM REFERENCE POLICY: {raw_logits} ; SUM: {raw_logits.sum()}")

        logprob = torch.gather(logits, 2, x['input_ids'][:, 1:].unsqueeze(-1)).squeeze(-1)
        logprob[~loss_mask[:, 1:].to(torch.bool)] = 0
        out = logprob.sum(-1)

        return out