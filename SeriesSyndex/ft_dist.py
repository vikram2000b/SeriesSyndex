import numpy as np
from torch.utils.data import DataLoader
import ot

class FTDistEvaluator:
    def __init__(self, real_dataset, logger, debug_logger, num_workers = 2, batch_size = 256):
        '''
        Contains fucntions to evaluate closeness to real data after fourier transform
        Args:
            real_dataset (torch.utils.data.Dataset): The real dataset which is used to evaluate the given syntehtic data.
            num_workers: number of CPU processes to use for loading the dataset
            batch_size: batch_size for loading the data, adjust according to your machine
        '''
        self.logger = logger
        self.debug_logger = debug_logger
        self.real_dataset = real_dataset
        self.num_workers = num_workers
        self.batch_size = batch_size

    def get_ft_wass_dist(self, synthetic_dataset, num_eval_batches=1):
        '''
        Fucntion to get wasserstein distances between fourier transforms of real and synthetic data
        Args:
            synthetic_dataset (torch.utils.data.Dataset): The synthetic dataset to be evaluated.
            num_eval_batches: number of batches to use for Fourier Transform and then distance measuremennt
        Returns:
            np.array: Wasserstein Distances for each feature
        '''
        real_loader = DataLoader(self.real_dataset, #num_workers=self.num_workers, 
                                 batch_size=self.batch_size)
        syn_loader = DataLoader(synthetic_dataset, #num_workers=self.num_workers,
                                batch_size=self.batch_size)
        
        eval_batch_num = 0
        real_eval_batches = []
        syn_eval_batches = []

        running_wass_dist_mean = None
        num_samples = None

        for real_batch, syn_batch in zip(real_loader, syn_loader):
            if  eval_batch_num < num_eval_batches:
                #accumulate temporal variable batches for evaluation
                eval_batch_num += 1
                real_eval_batches.append(real_batch[1].numpy())
                syn_eval_batches.append(syn_batch[1].numpy())
                eval_batch_num += 1
            else:
                real_eval_batch = np.concatenate(real_eval_batches)
                syn_eval_batch = np.concatenate(syn_eval_batches)

                real_eval_batch_ft = np.fft.fft(real_eval_batch, axis=1)
                syn_eval_batch_ft = np.fft.fft(syn_eval_batch, axis=1)

                wass_dist = np.zeros(real_eval_batch.shape[-1])

                for i in range(real_eval_batch.shape[-1]):
                    #separating the real and imaginary parts of fourier transformed ith temporal variable
                    real_eval_batch_real = np.real(real_eval_batch_ft[:, :, i])
                    real_eval_batch_imag = np.imag(real_eval_batch_ft[:, :, i])
#                     print(real_eval_batch_real.shape)

                    syn_eval_batch_real = np.real(syn_eval_batch_ft[:, :, i])
                    syn_eval_batch_imag = np.imag(syn_eval_batch_ft[:, :, i])

                    # Combine real and imaginary parts into a 2D array
                    distribution_real = np.column_stack((real_eval_batch_real.flatten(), real_eval_batch_imag.flatten()))
                    distribution_syn = np.column_stack((syn_eval_batch_real.flatten(), syn_eval_batch_imag.flatten()))

                    # Calculate the 2D Wasserstein distance
                    wass_dist[i] = ot.sliced_wasserstein_distance(distribution_real, distribution_syn)

                num_eval_batch_samples = real_eval_batch.shape[0]

                if running_wass_dist_mean is None:
                    running_wass_dist_mean = wass_dist
                    num_samples = num_eval_batch_samples
                else:
                    running_wass_dist_mean = (num_samples*running_wass_dist_mean + wass_dist)/(num_samples + num_eval_batch_samples)
                    num_samples += num_eval_batch_samples

                eval_batch_num = 0
                real_eval_batches = [real_batch[1].numpy()]
                syn_eval_batches = [syn_batch[1].numpy()]

        real_eval_batch = np.concatenate(real_eval_batches)
        syn_eval_batch = np.concatenate(syn_eval_batches)

        real_eval_batch_ft = np.fft.fft(real_eval_batch, axis=1)
        syn_eval_batch_ft = np.fft.fft(syn_eval_batch, axis=1)

        wass_dist = np.zeros(real_eval_batch.shape[-1])

        for i in range(real_eval_batch.shape[-1]):
            #separating the real and imaginary parts of fourier transformed ith temporal variable
            real_eval_batch_real = np.real(real_eval_batch_ft[:, :, i])
            real_eval_batch_imag = np.imag(real_eval_batch_ft[:, :, i])
#                     print(real_eval_batch_real.shape)

            syn_eval_batch_real = np.real(syn_eval_batch_ft[:, :, i])
            syn_eval_batch_imag = np.imag(syn_eval_batch_ft[:, :, i])

            # Combine real and imaginary parts into a 2D array
            distribution_real = np.column_stack((real_eval_batch_real.flatten(), real_eval_batch_imag.flatten()))
            distribution_syn = np.column_stack((syn_eval_batch_real.flatten(), syn_eval_batch_imag.flatten()))

            # Calculate the 2D Wasserstein distance
            wass_dist[i] = ot.sliced_wasserstein_distance(distribution_real, distribution_syn)

        num_eval_batch_samples = real_eval_batch.shape[0]

        if running_wass_dist_mean is None:
            running_wass_dist_mean = wass_dist
            num_samples = num_eval_batch_samples
        else:
            running_wass_dist_mean = (num_samples*running_wass_dist_mean + wass_dist)/(num_samples + num_eval_batch_samples)
            num_samples += num_eval_batch_samples

        return running_wass_dist_mean
        
    def evaluate(self, synthetic_dataset, calib_params, num_eval_batches=1):
        '''
        Fucntion to get wasserstein distances between fourier transforms of real and synthetic data
        Args:
            synthetic_dataset (torch.utils.data.Dataset): The synthetic dataset to be evaluated.
            calib_params: hyperparameters to convert the distance into score
            num_eval_batches: number of batches to use for Fourier Transform and then distance measuremennt
        Returns:
            np.float: Evaluation Score
        '''
        self.debug_logger.info("Starting the Evaluate function of FT Dist.")
        wass_dist = self.get_ft_wass_dist(synthetic_dataset, num_eval_batches)
        self.debug_logger.debug(f"Wasserstein Distance: {wass_dist}")
        return np.mean(np.clip(calib_params/wass_dist, 0, 1))