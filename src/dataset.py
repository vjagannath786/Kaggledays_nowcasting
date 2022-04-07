import torch


class NocDataset:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        current_data = self.data[idx,:]
        current_target = self.targets[idx]

        return {
            "data": torch.tensor(current_data, dtype=torch.float),
            "target": torch.tensor(current_target, dtype=torch.int)
        }