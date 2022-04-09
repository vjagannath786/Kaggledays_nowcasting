import torch


class NocDataset:
    def __init__(self, F_matrix, data, targets, is_test=False):
        self.F_matrix = F_matrix
        self.data = data
        self.targets = targets
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        current_data = self.F_matrix[self.data[idx]]
        

        if self.is_test:

            return {
            "x":torch.tensor(current_data, dtype=torch.float),
            #"targets": None
                   }
        else:
            current_target = self.targets[idx]
            return {
            "x":torch.tensor(current_data, dtype=torch.float),
            "targets":torch.tensor(current_target, dtype=torch.float)
        }


        