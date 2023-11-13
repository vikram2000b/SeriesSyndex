from torch.utils.data import ConcatDataset, Dataset

class pMSEDataset(Dataset):
    def __init__(self, real_dataset, synthetic_dataset):
        super(pMSEDataset, self).__init__()
        self.real_dataset_len = len(real_dataset)
        self.synthetic_dataset_len = len(synthetic_dataset)
        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset

    def __getitem__(self, index):
        if(index < self.real_dataset_len):
            return self.real_dataset[index], 0.
        else:
            return self.synthetic_dataset[index-self.real_dataset_len], 1.
        
    def __len__(self):
        return self.real_dataset_len + self.synthetic_dataset_len
    
class MLEfficacyDataset(Dataset):
    def __init__(self, dataset, target_feature = 0):
        super(MLEfficacyDataset, self).__init__()
        self.dataset = dataset
        self.target_feature = target_feature
    
    def __getitem__(self, index):
        static_vars, series_vars = self.dataset[index]
        target = series_vars[-1, self.target_feature]
        return (static_vars, series_vars[:-1, :]), target
    
    def __len__(self):
        return len(self.dataset)
        