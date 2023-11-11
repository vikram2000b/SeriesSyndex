from SeriesSyndex.basic_stats import BasicStatsEvaluator

class Evaluator:
    def __init__(self, real_dataset):
        self.real_dataset = real_dataset

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
        #TODO
        pass


