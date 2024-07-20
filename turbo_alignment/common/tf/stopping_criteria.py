import torch
from transformers import PreTrainedTokenizerBase, StoppingCriteria


class EndTagCriteria(StoppingCriteria):
    def __init__(self, end_tag: str, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_tag = end_tag

    def __call__(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        assert input_ids.shape[0] == 1
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=False)[-len(self.stop_tag) :] == self.stop_tag
