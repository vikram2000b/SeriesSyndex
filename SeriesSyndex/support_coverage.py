import numpy as np
from torch.utils.data import DataLoader

class SupportCoverageEvaluator:
    def __init__(self, real_dataset, num_bins=20, num_workers = 2, batch_size = 256):
        '''
        Contains fucntions to evaluate closeness to real data in terms of number of samples in different numerical ranges
        Args:
            real_dataset (torch.utils.data.Dataset): The real dataset which is used to evaluate the given syntehtic data.
            num_bins: number of bins to divide the data into and compare the number of samples in each bin for real and synthetic data
            num_workers: number of CPU processes to use for loading the dataset
            batch_size: batch_size for loading the data, adjust according to your machine
        '''
        self.real_dataset = real_dataset
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.temporal_vars_min, self.temporal_vars_max = self.get_real_data_range()

        self.temporal_vars_cut_offs, self.real_data_coverage = self.get_real_data_coverage()

    def get_real_data_range(self):
        temp_temporal_vars_min, temp_temporal_vars_max = None, None

        real_loader = DataLoader(self.real_dataset, num_workers=self.num_workers, 
                                 batch_size=self.batch_size)
        
        for batch in real_loader:
            static_vars = batch[0].item()
            temporal_vars = batch[1].item()

            batch_temporal_vars_min = np.min(temporal_vars, (0, 1))
            if temp_temporal_vars_min is None:
                temp_temporal_vars_min = batch_temporal_vars_min
            else:
                temp_temporal_vars_min = np.minimum(temp_temporal_vars_min, batch_temporal_vars_min)

            batch_temporal_vars_max = np.max(temporal_vars, (0, 1))
            if temp_temporal_vars_max is None:
                temp_temporal_vars_max = batch_temporal_vars_max
            else:
                temp_temporal_vars_max = np.maximum(temp_temporal_vars_max, batch_temporal_vars_max)

        return temp_temporal_vars_min, temp_temporal_vars_max

    def get_real_data_coverage(self):
        real_loader = DataLoader(self.real_dataset, num_workers=self.num_workers, 
                                 batch_size=self.batch_size)
        
        temporal_vars_cut_offs = {}
        temporal_vars_counts = {}

        for i, (min_val, max_val) in enumerate(zip(self.temporal_vars_min, self.temporal_vars_max)):
            bin_cut_offs = np.linspace(min_val, max_val, num=21)

            # Store cut-offs in the dictionary
            temporal_vars_cut_offs[f'temporal_var_{i+1}'] = bin_cut_offs.tolist()

        for batch in real_loader:
            static_vars = batch[0].item()
            temporal_vars = batch[1].item()

            for i in range(len(temporal_vars_cut_offs.keys())):
                # Initialize counts for each bucket
                counts = [0] * 20

                temporal_var_i = temporal_vars[:, :, i]

                # Bin values into buckets
                binned_values = np.digitize(temporal_var_i, temporal_vars_cut_offs[f'temporal_var_{i+1}'], right=True)

                # Update counts
                unique, counts_per_bucket = np.unique(binned_values, return_counts=True)
                counts[unique - 1] += counts_per_bucket

                temporal_vars_counts[f'temporal_var_{i}'] = counts

        return temporal_vars_cut_offs, temporal_vars_counts

    def evaluate(self, synthetic_dataset):
        scaling_factor = len(self.real_dataset)/len(synthetic_dataset)

        syn_loader = DataLoader(synthetic_dataset, num_workers=self.num_workers,
                                batch_size=self.batch_size)    

        temporal_vars_counts = {}

        for batch in syn_loader:
            static_vars = batch[0].item()
            temporal_vars = batch[1].item()

            for i in range(len(self.temporal_vars_cut_offs.keys())):
                # Initialize counts for each bucket
                counts = [0] * 20

                temporal_var_i = temporal_vars[:, :, i]

                # Bin values into buckets
                binned_values = np.digitize(temporal_var_i, self.temporal_vars_cut_offs[f'temporal_var_{i+1}'], right=True)

                # Update counts
                unique, counts_per_bucket = np.unique(binned_values, return_counts=True)
                counts[unique - 1] += counts_per_bucket

                temporal_vars_counts[f'temporal_var_{i}'] = counts 

        temporal_vars_coverage = []

        for i in range(len(temporal_vars_counts.keys())):
            temp_var_i_coverage = np.array(self.real_data_coverage[[f'temporal_var_{i}']])/(np.array(self.temporal_vars_counts[[f'temporal_var_{i}']])+1e-6)
            temp_var_i_coverage *= scaling_factor
            temp_var_i_coverage = np.mean(np.clip(temp_var_i_coverage, 0,  2))

            '''
            2 is the maximum score allowed for a bin, and is most sensitive to number of samples in rare bins 
            for the synthetic data, should be set according to senstivity requirements for rare bins i.e. bins
            with less number of samples in real data
            '''

            temporal_vars_coverage.append(temp_var_i_coverage)

        return np.mean(np.clip(np.array(temporal_vars_coverage), 0, 1))