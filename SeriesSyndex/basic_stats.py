import numpy as np
from torch.utils.data import DataLoader

class BasicStatsEvaluator:
    def __init__(self, real_dataset, num_workers = 1, batch_size = 256):
        '''
        Contains fucntions to evaluate closeness to real data in terms of baisc stats like mean, median, and standard deviation
        Args:
            real_dataset (torch.utils.data.Dataset): The real dataset which is used to evaluate the given syntehtic data.
            num_workers: number of CPU processes to use for loading the dataset
            batch_size: batch_size for loading the data, adjust according to your machine
        '''
        self.real_dataset = real_dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
    
    def mape (self, vector_a, vector_b):
        return abs(vector_a-vector_b)/abs(vector_a+1e-6)

    def evaluate(self, synthetic_dataset):
        # Not specifying num_workers due to issues in Windows Machine
        real_loader = DataLoader(self.real_dataset, batch_size=self.batch_size)
        syn_loader = DataLoader(synthetic_dataset, batch_size=self.batch_size)
        print("Evaluating Mean & Std of Real Data.")
        real_mean, real_std = self.get_mean_std(real_loader)
        print("Evaluating Mean & Std of Synthetic Data.")
        syn_mean, syn_std = self.get_mean_std(syn_loader)
        mean_mape = np.clip(self.mape(real_mean, syn_mean), 0, 1)
        score = np.sum(mean_mape)

        std_mape = np.clip(self.mape(real_std, syn_std), 0, 1)
        score += np.sum(std_mape)

        # TODO - Cache the mean & std of real data after first execution
        
        # median_mape = np.clip(mape(real_median, fake_median), 0, 1)
        # score += np.sum(median_mape)
        score /= len(real_mean) +len(real_std) #+ len(real_median)

        score = 1-score if score<=1.0 else 0.0
        #print('1:', score)
        return score

    def get_mean_std(self, dataloader):
        '''
        Function to get mean and standard deviation of the attributes of a dataset.
        Args:
            dataloader (pytorch DataLoader): The dataloader of the dataset for which the means are required.
        
        Returns:
            np.array: Mean value of each attribute
        '''
        running_mean = None
        running_mean_sq = None
        num_samples = 0
        
        for batch in dataloader:
            static_vars = batch[0].numpy()
            temporal_vars = batch[1].numpy()
            temporal_vars_sq = np.square(temporal_vars)

            num_batch_samples = static_vars.shape[0]

            batch_mean = np.mean(temporal_vars, (0, 1))
            batch_mean_sq = np.mean(temporal_vars_sq, (0, 1))

            if running_mean is None:
                running_mean = batch_mean
                running_mean_sq = batch_mean
            else:
                #TO DO - Solve the problem of overflow in larger datasets
                running_mean = (num_samples*running_mean + batch_mean)/(num_samples+num_batch_samples)
                running_mean_sq = (num_samples*running_mean_sq + batch_mean_sq)/(num_samples+num_batch_samples)
            
            num_samples += num_batch_samples
            # print(batch)
        
        final_mean = running_mean
        final_std = np.sqrt(running_mean_sq - np.square(running_mean))

        return final_mean, final_std
