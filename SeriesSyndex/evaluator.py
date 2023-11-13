from SeriesSyndex.basic_stats import BasicStatsEvaluator
from SeriesSyndex.pmse import pMSEEvaluator
from SeriesSyndex.ml_efficacy import MLEfficacyEvaluator
from SeriesSyndex.ft_dist import FTDistEvaluator

from SeriesSyndex.logger import setup_logger
import logging

logger = setup_logger("run.log", level = logging.INFO)
debug_logger = setup_logger("debug.log", level = logging.DEBUG)

import os
print(os.getcwd())

class Evaluator:
    def __init__(self, real_dataset, num_features):
        logger.info("Initiating the Evaluator Class.")
        debug_logger.info("Initiating the Evaluator Class.")
        self.real_dataset = real_dataset
        self.num_features = num_features
        debug_logger.info(f"Number of features in datasets: {self.num_features}")
        logger.info("Creating the Basic Statistics Evaluator")
        self.stats_evaluator = BasicStatsEvaluator(real_dataset)
        logger.info("Creating the pMSE Evaluator")
        self.pmse_evaluator = pMSEEvaluator(real_dataset, num_features=self.num_features, logger=logger, debug_logger=debug_logger)
        logger.info("Creating the ML Efficacy Evaluator")
        self.ml_eff_evaluator = MLEfficacyEvaluator(real_dataset, num_feature=self.num_features, logger=logger, debug_logger=debug_logger)
        self.ft_disct_evaluator = FTDistEvaluator(real_dataset=real_dataset)

    def calibrate(self, portion_size = 0.2, num_pairs = 1):
        '''
        Method to calibrate the parameters of the evaluator for the optimal evaluation of the metric.
        Args:
            portion_size (float): The size of each sampled portion of dataset. 
            num_pairs (int): The number of pairs to create for calibration.

        Returns:
        '''
        debug_logger.info(f"Calibrating the EValuator")
        #TODO
        pass

    def evaluate(self, synthetic_data):
        
        overall_score = 0
        stats_score = self.stats_evaluator.evaluate(synthetic_data)
        overall_score += stats_score

        ml_eff_score = self.ml_eff_evaluator.evaluate(synthetic_data)
        overall_score += ml_eff_score

        pmse_score = self.pmse_evaluator.evaluate(synthetic_data)
        overall_score += pmse_score

        ft_dist_score = self.ft_disct_evaluator(synthetic_data)

        
        # TODO
        return {
            "score": overall_score,
            "basic_stats_score": stats_score,
            "pmse_score": pmse_score,
            "ml_eff_score": ml_eff_score,
            "ft_disct_score": ft_dist_score
        }


