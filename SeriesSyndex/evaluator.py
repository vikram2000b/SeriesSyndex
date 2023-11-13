from SeriesSyndex.basic_stats import BasicStatsEvaluator
from SeriesSyndex.pmse import pMSEEvaluator
from SeriesSyndex.ml_efficacy import MLEfficacyEvaluator

from SeriesSyndex.logger import setup_logger
import logging

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
        log.info("Creating the Basic Statistics Evaluator")
        self.stats_evaluator = BasicStatsEvaluator(real_dataset)
        log.info("Creating the pMSE Evaluator")
        self.pmse_evaluator = pMSEEvaluator(real_dataset, num_features=self.num_features)
        log.info("Creating the ML Efficacy Evaluator")
        self.ml_eff_evaluator = MLEfficacyEvaluator(real_dataset, num_feature=self.num_features)

    def calibrate(self, portion_size = 0.2, num_pairs = 1):
        '''
        Method to calibrate the parameters of the evaluator for the optimal evaluation of the metric.
        Args:
            portion_size (float): The size of each sampled portion of dataset. 
            num_pairs (int): The number of pairs to create for calibration.

        Returns:
        '''
        debug_log(f"Calibrating the function")
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

        
        # TODO
        return {
            "score": overall_score,
            "basic_stats_score": stats_score,
            "pmse_score": pmse_score,
            "ml_eff_score": ml_eff_score
        }


