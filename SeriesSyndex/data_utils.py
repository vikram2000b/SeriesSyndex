from torch.utils.data import ConcatDataset, Dataset

class pMSEDataset(Dataset):
    def __init__(self, real_dataset, synthetic_dataset):
        super().__init__()
        self.real_dataset_len = len(real_dataset)
        self.synthetic_dataset_len = len(synthetic_dataset)
        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset

    def __getitem__(self, index):
        if(index < self.real_dataset_len):
            return self.real_dataset[index], 0
        else:
            return self.synthetic_dataset[index], 1
        