from SeriesSyndex.basic_stats import BasicStatsEvaluator
from SeriesSyndex.pmse import pMSEEvaluator

class Evaluator:
    def __init__(self, real_dataset, num_features):
        self.real_dataset = real_dataset
        self.num_features = num_features

        self.stats_evaluator = BasicStatsEvaluator(real_dataset)
        self.pmse_evaluator = pMSEEvaluator(real_dataset, num_features=self.num_features)

    def calibrate(self, portion_size = 0.2, num_pairs = 1):
        '''
        Method to calibrate the parameters of the evaluator for the optimal evaluation of the metric.
        Args:
            portion_size (float): The size of each sampled portion of dataset. 
            num_pairs (int): The number of pairs to create for calibration.

        Returns:
        '''
        #TODO
        pass

    def evaluate(self, synthetic_data):
        
        overall_score = 0
        stats_score = self.stats_evaluator(synthetic_data)
        overall_score += stats_score

        pmse_score = self.pmse_evaluator(synthetic_data)
        overall_score += pmse_score

        # TODO
        return {
            "score": overall_score,
            "basic_stats_score": stats_score,
            "pmse_score": pmse_score
        }


