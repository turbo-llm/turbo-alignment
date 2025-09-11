import dataclasses

from transformers.training_args import TrainingArguments

from turbo_alignment.modeling.parallel_states import initialize_model_parallel


@dataclasses.dataclass
class TrainingArgumentsWithSeqP(TrainingArguments):
    sequence_parallel: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.sequence_parallel > 1:
            initialize_model_parallel(sequence_parallel_size=self.sequence_parallel)
