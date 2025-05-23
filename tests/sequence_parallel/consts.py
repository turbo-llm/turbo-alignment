GEMMA_MODEL_PATH = '/mnt/models/google/gemma2-2b'
QWEN_MODEL_PATH = '/mnt/models/Qwen2.5-1.5B-Instruct/'
QWEN3_MODEL_PATH = '/mnt/models/Qwen3-0.6B'


DEEPSPEED_CONFIG = {
    "fp16": {
        "enabled": False,
    },
    "bf16": {
        "enabled": True,
    },
    # "optimizer": {
    #     "type": "AdamW",
    #     "params": {
    #         "lr": 0.003,
    #         "betas": [0.1, 0.2],
    #         "eps": 1e-8,
    #         "weight_decay": 0.1
    #     }
    # },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0.1,
            "warmup_max_lr": 0.1,
            "warmup_num_steps": 1,
            "total_num_steps": 100,
        },
    },
    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": True,
        "allgather_bucket_size": 1e5,
        "overlap_comm": False,
        "reduce_scatter": True,
        "reduce_bucket_size": "auto",
        "contiguous_gradients": True,
        "sub_group_size": 1e5,
        "stage3_param_persistence_threshold": "auto",
        "stage3_prefetch_bucket_size": 0,
        "stage3_max_live_parameters": 0,
        "stage3_max_reuse_distance": 0,
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "gradient_accumulation_steps": 1,
    "gradient_clipping": "auto",
    "steps_per_print": 2,
    "train_batch_size": 1,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False,
}
