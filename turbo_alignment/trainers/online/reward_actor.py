import ray
from turbo_alignment.trainers.online.ray.distributed_torch_ray_actor import DistributedTorchRayActor
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

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
            num_labels=1, #FIXME
            device_map='cuda', 
            torch_dtype=torch.bfloat16, 
            attn_implementation='flash_attention_2',
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            rm_model, 
            trust_remote_code=True, # FIXME!!!
        )
        
        # self.model.config.pad_token_id = self.tokenizer.encode('<|endoftext|>', add_special_tokens=False) #self.model.config.pad_token_id # FIXME
        self.model.config.pad_token_id = 151643 #FIXME
        self.tokenizer.pad_token = '<|endoftext|>' #FIXME
        # self.tokenizer.pad_token = self.tokenizer.encode('<|endoftext|>', add_special_tokens=False) # FIXME
        # self.tokenizer.pad_token = self.tokenizer.eos_token # FIXME

        # print(self.tokenizer.pad_token)
        # print(self.model.config.pad_token_id)

        # print(f"Reward model initialized on Node {self.node_id}, Local Rank {self.local_rank}")
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
        torch.cuda.empty_cache()
        self.model.eval()
        x = {k: v.cuda() for k, v in x.items()}

        # for input_ids in x['input_ids']:
        #     x['input_ids'] = input_ids

        # x.pop('position_ids') # FIXME OR NOT?

        # print('BEFORE!!!, ', x.keys(), x['input_ids'].shape, x['attention_mask'].shape)

        eoses = self.tokenizer.encode('\n<|im_end|>', return_tensors='pt', add_special_tokens=False).repeat(x['input_ids'].size(0), 1).cuda()
        eoses_attn_mask = torch.ones(eoses.shape).cuda()

        x['input_ids'] = torch.concat((x['input_ids'], eoses), dim=1)
        x['attention_mask'] = torch.concat((x['attention_mask'], eoses_attn_mask), dim=1)

        # x['input_ids'].shape()
        # x['attention_mask'].shape()

        # x['input_ids'] = torch.concat()

        # print('AFTER!!!, ', x.keys(), x['input_ids'].shape, x['attention_mask'].shape)

        # print('REWARD INPUTS INSIDE RM ', eoses, eoses.shape, x['input_ids'].shape,  x['input_ids'])

        # TODO CHECK IS ALL CORRECT WITH PADDINGS

        # print(self.tokenizer.pad_token)
        # print(self.model.config.pad_token_id)
        # print('\n\n\nRM model', [v.shape for k, v in x.items()], 'RM model\n\n\n')
        # print(self.tokenizer.decode(x['input_ids'][0], skip_special_tokens=False))
        
        # for k, v in x.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f'REWARD MODEL:{v.shape=}', flush=True)

        return self.model(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
        ).logits