import numpy as np
from torch.utils.data import DataLoader
import ot

class FTDistEvaluator:
    def __init__(self, real_dataset, logger, debug_logger, num_workers = 2, batch_size = 256, max_batches=None):
        '''
        Contains fucntions to evaluate closeness to real data after fourier transform
        Args:
            real_dataset (torch.utils.data.Dataset): The real dataset which is used to evaluate the given syntehtic data.
            num_workers (int): number of CPU processes to use for loading the dataset
            batch_size (int): batch_size for loading the data, adjust according to your machine
            max_batches (int): max_batches to use for score calculation for both the real and synthetic data 
            logger (logging.Logger): logger for basic logging
            debug_logger (logging.Logger): logger for more exhaustive debug logging
        '''
        self.logger = logger
        self.debug_logger = debug_logger
        self.real_dataset = real_dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.max_batches = max_batches

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

        num_batches_processed = 0
        self.debug_logger.debug(f"Maximum Number of Batches: {self.max_batches}")
        for i, (real_batch, syn_batch) in enumerate(zip(real_loader, syn_loader)):
            self.debug_logger.debug(f"Number of Batches Processed {num_batches_processed}")
            self.debug_logger.debug(f"Processing batch {i+1}")
            
            if  eval_batch_num < num_eval_batches:
                #accumulate temporal variable batches for evaluation
                eval_batch_num += 1
                real_eval_batches.append(real_batch[1].numpy())
                syn_eval_batches.append(syn_batch[1].numpy())
                eval_batch_num += 1
                num_batches_processed += 1

                if self.max_batches and (num_batches_processed>=self.max_batches):
                    self.debug_logger.debug("Stopping the loop and max batches limit reached.")
                    break
            else:
                real_eval_batch = np.concatenate(real_eval_batches)
                syn_eval_batch = np.concatenate(syn_eval_batches)

                self.debug_logger.info(f"Collected eval batch")
                self.debug_logger.info(f"Calculating FFT")
                real_eval_batch_ft = np.fft.fft(real_eval_batch, axis=1)
                syn_eval_batch_ft = np.fft.fft(syn_eval_batch, axis=1)

                #Picking only the top n_components of Fourier Transformed data
                n_components = min(real_eval_batch.shape[1]//2, 50)
                real_top_n_indices = np.argsort(np.abs(real_eval_batch_ft), axis=1)[:, -n_components:, :]
                real_eval_batch = np.take_along_axis(real_eval_batch, real_top_n_indices, axis = 1)
                # print(real_eval_batch.shape)
                syn_top_n_indices = np.argsort(np.abs(syn_eval_batch_ft), axis=1)[:, -n_components:, :]
                syn_eval_batch = np.take_along_axis(syn_eval_batch, syn_top_n_indices, axis = 1)
                # print(syn_eval_batch.shape)
                wass_dist = np.zeros(real_eval_batch.shape[-1])

                for feat_num in range(real_eval_batch.shape[-1]):
                    #separating the real and imaginary parts of fourier transformed ith temporal variable
                    real_eval_batch_real = np.real(real_eval_batch_ft[:, :, feat_num])
                    real_eval_batch_imag = np.imag(real_eval_batch_ft[:, :, feat_num])
                    # print(real_eval_batch_real.shape)

                    syn_eval_batch_real = np.real(syn_eval_batch_ft[:, :, feat_num])
                    syn_eval_batch_imag = np.imag(syn_eval_batch_ft[:, :, feat_num])

                    # Combine real and imaginary parts into a 2D array
                    distribution_real = np.column_stack((real_eval_batch_real.flatten(), real_eval_batch_imag.flatten()))
                    distribution_syn = np.column_stack((syn_eval_batch_real.flatten(), syn_eval_batch_imag.flatten()))

                    # Calculate the 2D Wasserstein distance
                    self.debug_logger.info(f"Calculating Wasserstein Distance")
                    wass_dist[feat_num] = ot.sliced_wasserstein_distance(distribution_real, distribution_syn, n_projections=20)

                num_eval_batch_samples = real_eval_batch.shape[0]

                if running_wass_dist_mean is None:
                    running_wass_dist_mean = wass_dist
                    num_samples = num_eval_batch_samples
                else:
                    running_wass_dist_mean = (num_samples*running_wass_dist_mean + wass_dist)/(num_samples + num_eval_batch_samples)
                    num_samples += num_eval_batch_samples

                eval_batch_num = 1
                real_eval_batches = [real_batch[1].numpy()]
                syn_eval_batches = [syn_batch[1].numpy()]

                num_batches_processed += 1
                if self.max_batches and (num_batches_processed>=self.max_batches):
                    break

        real_eval_batch = np.concatenate(real_eval_batches)
        syn_eval_batch = np.concatenate(syn_eval_batches)

        self.debug_logger.info(f"Collected eval batch")
        self.debug_logger.info(f"Calculating FFT")
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
            self.debug_logger.info(f"Calculating Wasserstein Distance")
            wass_dist[i] = ot.sliced_wasserstein_distance(distribution_real, distribution_syn, n_projections=20)

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
        self.logger.info("Starting the Evaluate function of FT Dist.")
        self.debug_logger.info("Starting the Evaluate function of FT Dist.")
        wass_dist = self.get_ft_wass_dist(synthetic_dataset, num_eval_batches)
        self.logger.debug(f"Wasserstein Distance: {wass_dist}")
        self.debug_logger.debug(f"Wasserstein Distance: {wass_dist}")
        return np.mean(np.clip(calib_params/wass_dist, 0, 1))