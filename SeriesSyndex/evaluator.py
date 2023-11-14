from SeriesSyndex.basic_stats import BasicStatsEvaluator
from SeriesSyndex.pmse import pMSEEvaluator
from SeriesSyndex.ml_efficacy import MLEfficacyEvaluator
from SeriesSyndex.support_coverage import SupportCoverageEvaluator
from SeriesSyndex.ft_dist import FTDistEvaluator
from torch.utils.data import Subset
from SeriesSyndex.logger import setup_logger
import logging
import numpy as np

logger = setup_logger("run.log", level = logging.INFO)
debug_logger = setup_logger("debug.log", level = logging.DEBUG)

class Evaluator:
    def __init__(self, real_dataset, num_features, batch_size = 256, target_feature = 0, max_batches = None):
        logger.info("Initiating the Evaluator Class.")
        debug_logger.info("Initiating the Evaluator Class.")
        self.real_dataset = real_dataset
        self.num_features = num_features
        self.target_feature = target_feature
        self.batch_size = batch_size
        self.max_batches = max_batches
        debug_logger.info(f"Number of features in datasets: {self.num_features}")

        logger.info("Calibrating the parameters")
        self.calib_params = self.calibrate()

        try:
            logger.info("Creating the Basic Statistics Evaluator")
            debug_logger.info("Creating the Basic Statistics Evaluator")
            self.stats_evaluator = BasicStatsEvaluator(real_dataset, logger=logger, debug_logger=debug_logger, 
                                                    batch_size=batch_size)
        except Exception as e:
            logger.info(f"Basic Stats Evaluator Creation Failed. Error: {str(e)}")
            debug_logger.debug(f"Basic Stats Evaluator Creation Failed. Error: {str(e)}")
            self.stats_evaluator = None

        try:
            logger.info("Creating the pMSE Evaluator")
            self.pmse_evaluator = pMSEEvaluator(real_dataset, num_features=self.num_features, logger=logger, 
                                                debug_logger=debug_logger, batch_size=batch_size)
        except Exception as e:
            logger.info(f"PMSE Evaluator Creation Failed. Error: {str(e)}")
            debug_logger.debug(f"PMSE Evaluator Creation Failed. Error: {str(e)}")
            self.pmse_evaluator = None

        try:
            logger.info("Creating the ML Efficacy Evaluator")
            self.ml_eff_evaluator = MLEfficacyEvaluator(real_dataset, num_features=self.num_features, 
                                                        logger=logger, debug_logger=debug_logger,
                                                        batch_size=batch_size)
        except Exception as e:
            logger.info(f"ML Efficacy Evaluator Creation Failed. Error: {str(e)}")
            debug_logger.debug(f"ML Efficacy Evaluator Creation Failed. Error: {str(e)}")
            self.ml_eff_evaluator = None
        
        try:
            logger.info("Creating the Support Coverage Evaluator")
            self.sup_cov_evaluator = SupportCoverageEvaluator(real_dataset, logger=logger, 
                                                              debug_logger=debug_logger, batch_size=batch_size)
        except Exception as e:
            logger.info(f"Support Coverage Creation Failed. Error: {str(e)}")
            debug_logger.debug(f"Support Coverage Creation Failed. Error: {str(e)}")
            self.sup_cov_evaluator = None

        try:
            logger.info("Creating the Fourier Transform Distance Evaluator")
            self.ft_dist_evaluator = FTDistEvaluator(real_dataset, logger=logger, debug_logger=debug_logger,
                                                     batch_size=batch_size)
        except Exception as e:
            logger.info(f"Fourier Transform Distance Evaluator Creation Failed. Error: {str(e)}")
            debug_logger.debug(f"Fourier Transform Distance Evaluator Creation Failed. Error: {str(e)}")
            self.ft_dist_evaluator = None


    def calibrate(self, portion_size = 0.2, num_pairs = 1):
        '''
        Method to calibrate the parameters of the evaluator for the optimal evaluation of the metric.
        Args:
            portion_size (float): The size of each sampled portion of dataset. 
            num_pairs (int): The number of pairs to create for calibration.

        Returns:
            dict: appropriate hyperparameters for different evaluators
        '''
        debug_logger.info(f"Calibrating the function")
        #TODO - Find appropriate base for PMSE
        
        #Calibrate FT_dist using distances between real data subsets
        subset_size = int(len(self.real_dataset)*portion_size)

        indices = list(range(len(self.real_dataset)))

        wass_dists = []

        for i in range(num_pairs):
            np.random.shuffle(indices)
            subset_indices_1 = indices[:subset_size]

            subset_1 = Subset(self.real_dataset, subset_indices_1)

            np.random.shuffle(indices)
            subset_indices_2 = indices[:subset_size]

            subset_2 = Subset(self.real_dataset, subset_indices_2)

            ft_dist_evaluator = FTDistEvaluator(subset_1, logger=logger, debug_logger=debug_logger)
            # print(ft_dist_evaluator.get_ft_wass_dist(subset_2))
            wass_dists.append(ft_dist_evaluator.get_ft_wass_dist(subset_2))

        return {'ft_dist_params': np.mean(np.array(wass_dists), axis=0)}

    def evaluate(self, synthetic_data, cat_cols=[]):
        #TO DO - Make each evaluation individual processes for multiprocessing
        overall_score = 0
        
        if self.stats_evaluator:
            try:
                stats_score = self.stats_evaluator.evaluate(synthetic_data, cat_cols)
                overall_score += stats_score
            except Exception as e:
                logger.info(f"Basic Statistics Calculation Failed!! Error: {str(e)}")
                debug_logger.info(f"Basic Statistics Calculation Failed!! Error: {str(e)}")
                stats_score = None

        if self.ml_eff_evaluator:
            try:
                ml_eff_score = self.ml_eff_evaluator.evaluate(synthetic_data)
                overall_score += ml_eff_score
            except Exception as e:
                logger.info(f"ML Efficacy Calculation Failed!! Error: {str(e)}")
                debug_logger.info(f"ML Efficacy Calculation Failed!! Error: {str(e)}")
                ml_eff_score = None

        if self.pmse_evaluator:
            try:
                pmse_score = self.pmse_evaluator.evaluate(synthetic_data)
                overall_score += pmse_score
            except Exception as e:
                logger.info(f"PMSE Calculation Failed!! Error: {str(e)}")
                debug_logger.info(f"PMSE Calculation Failed!! Error: {str(e)}")
                pmse_score = None
        if self.sup_cov_evaluator:
            try:
                sup_cov_score = self.sup_cov_evaluator.evaluate(synthetic_data)
                overall_score += sup_cov_score
            except Exception as e:
                logger.info(f"Support Coverage Calculation Failed!! Error: {str(e)}")
                debug_logger.info(f"Support Coverage Calculation Failed!! Error: {str(e)}")
                sup_cov_score = None

        if self.ft_dist_evaluator:
            try:
                ft_dist_score = self.ft_dist_evaluator.evaluate(synthetic_data, self.calib_params['ft_dist_params'])
                overall_score += ft_dist_score
            except Exception as e:
                logger.info(f"Fourier Transform Dist Calculation Failed!! Error: {str(e)}")
                debug_logger.info(f"Fourier Transform Dist Calculation Failed!! Error: {str(e)}")
                ft_dist_score = None

       
        return {
            "score": overall_score,
            "basic_stats_score": stats_score,
            "pmse_score": pmse_score,
            "ml_eff_score": ml_eff_score,
            "sup_cov_score": sup_cov_score,
            "ft_dist_score": ft_dist_score
        }


