import numpy as np
from torch.utils.data import DataLoader
from dython.nominal import associations

class BasicStatsEvaluator:
    def __init__(self, real_dataset, logger, debug_logger, num_workers = 1, batch_size = 256, max_batches = None):
        '''
        Contains fucntions to evaluate closeness to real data in terms of baisc stats like mean, median, and standard deviation
        Args:
            real_dataset (torch.utils.data.Dataset): The real dataset which is used to evaluate the given syntehtic data.
            num_workers: number of CPU processes to use for loading the dataset
            batch_size: batch_size for loading the data, adjust according to your machine
        '''

        self.debug_logger = debug_logger
        self.debug_logger.info("Initiating the Basic Statistics Evaluator.")
        self.logger = logger
        self.real_dataset = real_dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.max_batches = max_batches
    
    def mape (self, vector_a, vector_b):
        return abs(vector_a-vector_b)/abs(vector_a+1e-6)

    def log_corr(self, data, cat_cols=[]):
        corr = associations(data, nominal_columns=cat_cols, nom_nom_assoc='theil', multiprocessing=True, plot=False)['corr'].to_numpy()
        corr_log = np.sign(corr)*np.log(abs(corr))
        return corr_log 

    def get_mean_std_corr(self, dataloader, cat_cols=[]):
        '''
        Function to get mean and standard deviation of the attributes of a dataset.
        Args:
            dataloader (pytorch DataLoader): The dataloader of the dataset for which the means are required.
        
        Returns:
            np.array: Mean value of each attribute
        '''
        self.debug_logger.info("Getting Mean and Standard Deviation of the data.")
        running_mean = None
        running_mean_sq = None
        running_log_corr = None
        num_samples = 0
        num_batches_processed = 0
        for batch in dataloader:
            static_vars = batch[0].numpy()
            temporal_vars = batch[1].numpy()
            temporal_vars_sq = np.square(temporal_vars)

            num_batch_samples = static_vars.shape[0]

            batch_mean = np.mean(temporal_vars, (0, 1))
            batch_mean_sq = np.mean(temporal_vars_sq, (0, 1))
            batch_log_corr = self.log_corr(temporal_vars.reshape((-1, temporal_vars.shape[-1])), cat_cols)

            if running_mean is None:
                running_mean = batch_mean
                running_mean_sq = batch_mean_sq
                if temporal_vars.shape[-1] > 1:
                    running_log_corr = batch_log_corr
            else:
                #TO DO - Solve the problem of overflow in larger datasets
                running_mean = (num_samples*running_mean + batch_mean)/(num_samples+num_batch_samples)
                running_mean_sq = (num_samples*running_mean_sq + batch_mean_sq)/(num_samples+num_batch_samples)
                if temporal_vars.shape[-1] > 1:
                    running_log_corr = (num_samples*running_log_corr + batch_log_corr)/(num_samples+num_batch_samples)
            
            num_samples += num_batch_samples
            num_batches_processed += 1
            if self.max_batches and (num_batches_processed >= self.max_batches):
                break
            # print(batch)

        final_std = np.sqrt(running_mean_sq - np.square(running_mean))

        return running_mean, final_std, running_log_corr   

    def evaluate(self, synthetic_dataset, cat_cols=[]):
        # Not specifying num_workers due to issues in Windows Machine
        real_loader = DataLoader(self.real_dataset, batch_size=self.batch_size)
        syn_loader = DataLoader(synthetic_dataset, batch_size=self.batch_size)

        self.logger.info("Evaluating Mean & Std & Correlation Matrix of Real Data.")
        self.debug_logger.info("Evaluating Mean & Std & Correlation Matrix of Real Data.")

        real_mean, real_std, real_log_corr = self.get_mean_std_corr(real_loader, cat_cols)

        self.logger.info(f"Mean: {real_mean} \n Standard Deviation: {real_std} \n Log Correlation: {real_log_corr}")
        self.debug_logger.debug(f"Mean: {real_mean} \n Standard Deviation: {real_std} \n Log Correlation: {real_log_corr}")

        self.logger.info("Evaluating Mean & Std & Correlation Matrix of Synthetic Data.")
        self.debug_logger.info("Evaluating Mean & Std & Correlation Matrix of Synthetic Data.")

        syn_mean, syn_std, syn_log_corr = self.get_mean_std_corr(syn_loader, cat_cols)

        self.logger.info(f"Mean: {real_mean} \n Standard Deviation: {real_std} \n Log Correlation: {real_log_corr}")
        self.debug_logger.debug(f"Mean: {real_mean} \n Standard Deviation: {real_std} \n Log Correlation: {real_log_corr}")

        mean_mape = np.clip(self.mape(real_mean, syn_mean), 0, 1)

        self.debug_logger.debug(f"Mean Mape: {mean_mape}")
        
        score = np.sum(mean_mape)/len(real_mean)

        self.logger.info(f"Mean Score: {1 - score}")
        self.debug_logger.debug(f"Mean Score: {1 - score}")

        std_mape = np.clip(self.mape(real_std, syn_std), 0, 1)

        self.debug_logger.debug(f"Std Mape: {std_mape}")

        score += np.sum(std_mape)/len(real_std)

        self.logger.info(f"Std Score: {1 - np.sum(std_mape)/len(real_std)}")
        self.debug_logger.debug(f"Std Score: {1 - np.sum(std_mape)/len(real_std)}")

        if real_log_corr is not None:
            corr_mape = np.sum(np.clip(self.mape(real_log_corr, syn_log_corr).flatten(), 0, 1))
            self.debug_logger.debug(f"Corr Mape: {corr_mape}")

            n = real_log_corr.shape[-1]
            self.debug_logger.debug(f"Corr Shape: {n}")

            corr_mape /= n**2 - n
            score += corr_mape
            self.logger.info(f"Correlation Score: {corr_mape}")
            self.debug_logger.debug(f"Correlation Score: {corr_mape}")
        else:
            self.logger.info(f"Correlation Score: NA")
            self.debug_logger.debug(f"Correlation Score: NA as the data contains only 1 feature")
        # score = 1-score if score<=1.0 else 0.0

        # TODO - Cache the mean, std & log correlation matrix of real data after first execution
        
        # median_mape = np.clip(mape(real_median, fake_median), 0, 1)
        # score += np.sum(median_mape)
        # score /= 3 #+ len(real_median)

        if real_log_corr is not None:
            score /= 3
        else:
            score /= 2

        score = 1-score if score<=1.0 else 0.0
        #print('1:', score)
        return score