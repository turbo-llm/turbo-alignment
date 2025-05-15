from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, records):
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int):
        return self._records[index]
