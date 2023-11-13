from SeriesSyndex.basic_stats import BasicStatsEvaluator
from SeriesSyndex.pmse import pMSEEvaluator
from SeriesSyndex.ml_efficacy import MLEfficacyEvaluator
from SeriesSyndex.support_coverage import SupportCoverageEvaluator
from SeriesSyndex.ft_dist import FTDistEvaluator
from torch.utils.data import Subset
from SeriesSyndex.logger import setup_logger
import logging
import numpy as np

log = setup_logger("run.log", level = logging.INFO)
debug_log = setup_logger("debug.log", level = logging.DEBUG)

import os
print(os.getcwd())

class Evaluator:
    def __init__(self, real_dataset, num_features):
        log.info("Initiating the Evaluator Class.")
        debug_log.info("Initiating the Evaluator Class.")
        self.real_dataset = real_dataset
        self.num_features = num_features
        debug_log.info(f"Number of features in datasets: {self.num_features}")

        log.info("Calibrating")
        self.calib_params = self.calibrate()

        log.info("Creating the Basic Statistics Evaluator")
        self.stats_evaluator = BasicStatsEvaluator(real_dataset)
        log.info("Creating the pMSE Evaluator")
        self.pmse_evaluator = pMSEEvaluator(real_dataset, num_features=self.num_features)
        log.info("Creating the ML Efficacy Evaluator")
        self.ml_eff_evaluator = MLEfficacyEvaluator(real_dataset, num_feature=self.num_features)
        log.info("Creating the Support Coverage Evaluator")
        self.sup_cov_evaluator = SupportCoverageEvaluator(real_dataset)
        log.info("Creating the Fourier Transform Distance Evaluator")
        self.ft_dist_evaluator = FTDistEvaluator(real_dataset)

    def calibrate(self, portion_size = 0.2, num_pairs = 1):
        '''
        Method to calibrate the parameters of the evaluator for the optimal evaluation of the metric.
        Args:
            portion_size (float): The size of each sampled portion of dataset. 
            num_pairs (int): The number of pairs to create for calibration.

        Returns:
            dict: appropriate hyperparameters for different evaluators
        '''
        debug_log.info(f"Calibrating the function")
        #TO DO - Find appropriate base for PMSE
        
        #Calibrate FT_dist using distances between real data subsets
        subset_size = int(len(self.real_dataset)*portion_size)

        indices = list(range(len(self.real_dataset)))

        wass_dists = []

        for i in range(num_pairs):
            shuffled_indices_1 = np.random.shuffle(indices)
            subset_indices_1 = indices[:subset_size]

            subset_1 = Subset(self.real_dataset, subset_indices_1)

            shuffled_indices_2 = np.random.shuffle(indices)
            subset_indices_2 = indices[:subset_size]

            subset_2 = Subset(self.real_dataset, subset_indices_2)

            ft_dist_evaluator = FTDistEvaluator(subset_1)
            # print(ft_dist_evaluator.get_ft_wass_dist(subset_2))
            wass_dists.append(ft_dist_evaluator.get_ft_wass_dist(subset_2))

        return {'ft_dist_params': np.mean(np.array(wass_dists), axis=0)}

    def evaluate(self, synthetic_data, cat_cols=[]):
        #TO DO - Make each evaluation individual processes for multiprocessing
        overall_score = 0
        stats_score = self.stats_evaluator.evaluate(synthetic_data, cat_cols)
        overall_score += stats_score

        ml_eff_score = self.ml_eff_evaluator.evaluate(synthetic_data)
        overall_score += ml_eff_score

        pmse_score = self.pmse_evaluator.evaluate(synthetic_data)
        overall_score += pmse_score

        sup_cov_score = self.sup_cov_evaluator.evaluate(synthetic_data)
        overall_score += sup_cov_score

        ft_dist_score = self.ft_dist_evaluator.evaluate(synthetic_data, self.calib_params['ft_dist_params'])
        overall_score += ft_dist_score
       
        return {
            "score": overall_score,
            "basic_stats_score": stats_score,
            "pmse_score": pmse_score,
            "ml_eff_score": ml_eff_score,
            "sup_cov_score": sup_cov_score,
            "ft_dist_score": ft_dist_score
        }


