import ray
from turbo_alignment.trainers.online.ray.distributed_torch_ray_actor import DistributedTorchRayActor
from transformers import AutoModelForCausalLM, AutoTokenizer

@ray.remote(num_gpus=1)
class ReferenceModel(DistributedTorchRayActor):
    def __init__(self, world_size, rank, local_rank, master_addr, master_port):
        super().__init__(world_size, rank, local_rank, master_addr, master_port)
        self.node_id = ray.get_runtime_context().get_node_id()
        self.local_rank = ray.get_gpu_ids()
    
    def init_model_from_pretrained(self, pretrain):
        self._setup_distributed()
        self.model = AutoModelForCausalLM.from_pretrained(pretrain, device_map='cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True)
        print(f"Reference model initialized on Node {self.node_id}, Local Rank {self.local_rank}")
        print("GPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["GPU"]))

    def tokenize(self, text: str):
        return self.tokenizer(text, return_tensors='pt')
    
    def generate(self, text: str):
        tokenized_input = self.tokenize(text).to('cuda')
        return self.model(**tokenized_input)
    
    def forward(self, x):
        return self.model(**x)