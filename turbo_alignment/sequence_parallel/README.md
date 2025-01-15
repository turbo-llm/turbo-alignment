# Ulysses attention

This directory contains code for implementing Ulysses Attention, as described in the [paper](https://arxiv.org/abs/2309.14509).
See also the DeepSpeed team's [blogpost](https://www.deepspeed.ai/tutorials/ds-sequence/).

## Using
Just set `sequence_parallel` in trainer settings and choose an adapted model.
Currently, only Gemma2 is supported, adpated version has name `gemma_with_ulysses`. An example of the settings file can be found in [tests](../../tests/fixtures/configs/train/dpo/dpo_with_seq_p.json).


## Implementation
1. We split all the workers into groups. Every worker in the group will process the same examples.
This is achieved by patching dataset sharding.
1. Each worker in the group process its own subsequence of example for all layers, except Attention.
1. In the attention layers, communication occurs as described in the paper.
1. We have to patch model graphs. We have to patch:
    1. Attention implementation class. Eager and Flash attention variants are implemented.
    1. Lengths arithmetic. Some code relies on input length, but since input_ids are now sharded across workers, adjustments are required.
    1. Positional embedding implementation: Updates are made to support sharded inputs.
1. Loss functions are adapted to account for the changes.
1. Parameters dependent on the number of workers (e.g., total batch size, number of steps) are adjusted accordingly.
1. We have to be sure, that all random generators of all workers inside group are perfectly synchronized.

## Tests
Tests for this code can be found in directory `tests/sequence_parallel`.
Most of them requires two GPUs and Gemma model.