# Choosing the Right `sequence_parallel` Value

This guide explains how to choose the optimal `sequence_parallel` (SP) setting in `trainer_settings` for maximum training throughput.

## TL;DR

> **Use the smallest SP value that fits your training in GPU memory.**

SP is a memory optimization, not a speed optimization. Every increase in SP degree adds communication overhead.

- Fits with SP=1? → Use **SP=1**
- OOM with SP=1, fits with SP=2? → Use **SP=2**
- OOM with SP=2, fits with SP=4? → Use **SP=4**
- And so on...

## Background: How Ulysses Sequence Parallelism Works

This codebase implements [Ulysses-style](https://arxiv.org/abs/2309.14509) sequence parallelism. When `sequence_parallel = SP` is set in trainer settings:

1. **GPU grouping**: The total `world_size` GPUs are divided into SP groups of `SP` GPUs each. The remaining dimension becomes data parallelism (DP):
   ```
   DP degree = world_size / SP
   ```

2. **Sequence splitting**: Each GPU in an SP group processes `1/SP` of the sequence length for all layers except attention.

3. **Attention communication**: In each attention layer, two `all_to_all` collectives are performed:
   - Before attention: scatter along heads, gather along sequence → each GPU gets the full sequence but a subset of attention heads
   - After attention: scatter along sequence, gather along heads → back to the original partitioning

4. **Gradient synchronization**: Standard DP gradient all-reduce happens across the `DP = world_size / SP` replicas.

## Why Higher SP Hurts Throughput

### Communication overhead scales with SP

Each attention layer requires 2 `all_to_all` operations. For a model with `L` layers, a full forward + backward pass involves `4L` all_to_all calls. The communication volume per GPU per call:

| SP | Fraction of tensor communicated per GPU | Peers |
|----|----------------------------------------|-------|
| 1  | 0% (no communication)                  | 0     |
| 2  | 50%                                    | 1     |
| 4  | 75%                                    | 3     |
| 8  | 87.5%                                  | 7     |

### Data parallelism decreases with SP

With fixed total GPU count `N`:

| SP | DP degree | Effective batch multiplier |
|----|-----------|--------------------------|
| 1  | N         | N                        |
| 2  | N/2       | N/2                      |
| 4  | N/4       | N/4                      |
| 8  | N/8       | N/8                      |

Higher SP means fewer DP replicas, fewer samples processed per step, and more optimization steps to finish training.

### Compute per GPU stays roughly constant

With fixed batch size `B` and sequence length `S`:
- SP=2: each GPU processes `B × S/2` tokens
- SP=4: each GPU processes `B × S/4` tokens

The "saved" compute is not free — it is replaced by communication overhead. With modern optimized attention kernels (flash attention), the compute savings are small relative to the communication cost.

## SP Group Placement and Node Topology

### Keep SP groups within a single node

The all_to_all communication within an SP group is latency-sensitive and bandwidth-intensive. The placement of SP groups relative to node boundaries has a dramatic effect on performance:

| SP group placement          | Interconnect      | Typical bandwidth          |
|-----------------------------|-------------------|----------------------------|
| All GPUs on same node       | NVLink / NVSwitch | 600–900 GB/s bidirectional |
| GPUs across 2 nodes         | Mix NVLink + IB   | ~100–150 GB/s effective    |
| GPUs across N nodes (1/node)| All InfiniBand    | ~50–100 GB/s per link      |

**Example with 8×H100 nodes**: SP groups are formed from consecutive ranks (`[0,1,2,3]`, `[4,5,6,7]`, etc.). With standard rank assignment (ranks 0–7 on node 0, 8–15 on node 1), any SP ≤ 8 keeps each SP group entirely within one node.

**Rule**: Choose SP ≤ GPUs-per-node to ensure all SP communication uses fast intra-node interconnect (NVSwitch).

## Scaling: More Nodes with Fixed SP

When SP groups fit within a single node, adding more nodes increases only DP degree. This is almost pure linear scaling:

| Setup (SP=4)             | Total GPUs | SP groups | DP degree | Effective batch |
|--------------------------|-----------|-----------|-----------|-----------------|
| 2 nodes × 8 H100        | 16        | 4         | 4         | bs × 4          |
| 4 nodes × 8 H100        | 32        | 8         | 8         | bs × 8          |
| 8 nodes × 8 H100        | 64        | 16        | 16        | bs × 16         |

Since SP all_to_all stays intra-node in all cases, adding nodes primarily adds DP gradient all-reduce overhead (which scales well with InfiniBand). **More nodes ≈ proportionally faster training.**

## Ulysses-Specific Constraints

- **Head divisibility**: `num_attention_heads` must be divisible by `SP`. For Qwen3-32B (64 heads), valid SP values are: 1, 2, 4, 8, 16, 32, 64.
- **Attention mask memory**: The 4D attention mask `[batch, 1, seq_len, seq_len]` is NOT split across SP ranks — every GPU in the SP group stores the full mask. This means the O(seq_len²) mask memory is not reduced by SP.
- **Sequence padding**: Sequences are padded to be divisible by SP. Small SP values waste less on padding.

## Configuration

The `sequence_parallel` parameter lives inside `trainer_settings` (default: `1`, i.e. disabled). When SP > 1, you **must** use a model type with Ulysses attention support.

### Available model types with Ulysses SP

| Model type string               | Base model  | Task                    |
|----------------------------------|-------------|-------------------------|
| `qwen3_with_ulysses`            | Qwen3       | Causal LM (SFT/DPO)    |
| `seq_cls_qwen3_with_ulysses`    | Qwen3       | Sequence Classification (RM) |
| `qwen_with_ulysses`             | Qwen2       | Causal LM               |
| `seq_cls_qwen_with_ulysses`     | Qwen2       | Sequence Classification  |
| `gemma_with_ulysses`            | Gemma2      | Causal LM               |
| `seq_cls_gemma_with_ulysses`    | Gemma2      | Sequence Classification  |

Standard model types (`causal`, `seq_cls`) do **not** support SP and must be used with `sequence_parallel: 1`.

### Example config (RM training with Qwen3-32B, SP=4)

```json
{
  "trainer_settings": {
    "sequence_parallel": 4,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "deepspeed": "deepspeed_configs/stage2.json",
    "bf16": true,
    "gradient_checkpointing": true,
    ...
  },
  "model_settings": {
    "model_path": "/path/to/Qwen3-32B",
    "model_type": "seq_cls_qwen3_with_ulysses",
    "transformers_settings": {
      "torch_dtype": "bfloat16"
    },
    "model_kwargs": {
      "num_labels": 1
    }
  },
  ...
}
```

> **Note**: Using a `*_with_ulysses` model type with `sequence_parallel: 1` also works — the Ulysses attention layers simply skip the all_to_all calls when SP is not initialized. However, there is no benefit to using the Ulysses model type without SP > 1.

## Decision Flowchart

```
Start
  │
  ▼
Does training fit in memory with SP=1?
  ├── Yes → Use SP=1 (maximum throughput)
  │
  ├── No
  │     │
  │     ▼
  │   Does it fit with SP=2?
  │     ├── Yes → Use SP=2
  │     │
  │     ├── No
  │     │     │
  │     │     ▼
  │     │   Does it fit with SP=4?
  │     │     ├── Yes → Use SP=4
  │     │     │
  │     │     ├── No → Continue increasing SP...
  │     │     │        (ensure num_heads % SP == 0)
  │     │     │        (ensure SP ≤ GPUs per node)
```

## Summary Table

| Factor                        | SP=1       | SP=2        | SP=4        | SP=8        |
|-------------------------------|------------|-------------|-------------|-------------|
| SP communication overhead     | None       | Low         | Moderate    | High        |
| DP degree (N GPUs)            | N          | N/2         | N/4         | N/8         |
| Per-GPU sequence memory       | Full       | 1/2         | 1/4         | 1/8         |
| Attention mask memory         | Full       | Full        | Full        | Full        |
| Max supported sequence length | 1× base    | 2× base     | 4× base     | 8× base     |
| Throughput (samples/sec)      | Highest    | High        | Moderate    | Lower       |

**Bottom line**: SP is a knob for trading throughput for the ability to handle longer sequences. Always use the minimum SP that fits in memory.
