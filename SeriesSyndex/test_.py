# Use PyTest run these test cases

from SeriesSyndex.basic_stats import BasicStatsEvaluator
from SeriesSyndex.pmse import pMSEEvaluator
from SeriesSyndex.ml_efficacy import MLEfficacyEvaluator
from SeriesSyndex.support_coverage import SupportCoverageEvaluator
from SeriesSyndex.logger import setup_logger
from SeriesSyndex.ft_dist import FTDistEvaluator
from torch.utils.data import Dataset
import numpy as np
import pytest
import logging

class TestDataset(Dataset):
    def __init__(self, series_x, static_x):
        super().__init__()
        
        assert series_x.shape[0] == static_x.shape[0]
        self.series_x = series_x
        self.static_x = static_x
        
    def __getitem__(self, index):
        return self.static_x[index], self.series_x[index]
    
    def __len__(self):
        return len(self.series_x)

@pytest.fixture
def init_vars():
	real_dataset_1000 = TestDataset(np.random.rand(1000, 100, 10), np.random.rand(1000, 100, 10))

	temporal_vars_cat_1000 = np.random.rand(1000, 100, 10)
	temporal_vars_cat_1000[:, :, 2:4] = np.random.randint(0, 10, (1000, 100, 2))
	real_dataset_cat_1000 = TestDataset(temporal_vars_cat_1000, np.random.rand(1000, 100, 10))

	real_dataset_100 = TestDataset(np.random.rand(100, 100, 10), np.random.rand(100, 100, 10))
	syn_dataset_1000 = TestDataset(np.random.rand(1000, 100, 10), np.random.rand(1000, 100, 10))

	temporal_vars_cat_1000 = np.random.rand(1000, 100, 10)
	temporal_vars_cat_1000[:, :, 2:4] = np.random.randint(0, 10, (1000, 100, 2))
	syn_dataset_cat_1000 = TestDataset(temporal_vars_cat_1000, np.random.rand(1000, 100, 10))

	syn_dataset_100 = TestDataset(np.random.rand(100, 100, 10), np.random.rand(100, 100, 10))
	real_dataset_zeros_1000 = TestDataset(np.zeros((1000, 100, 10)), np.zeros((1000, 100, 10)))

	logger = setup_logger("test_run.log", level = logging.INFO)
	debug_logger = setup_logger("test_debug.log", level = logging.DEBUG)

	return real_dataset_1000, real_dataset_cat_1000, real_dataset_zeros_1000, real_dataset_100, \
		syn_dataset_1000, syn_dataset_cat_1000, syn_dataset_100, logger, debug_logger

def test_basic_stats_equal(init_vars):
	stats_evaluator = BasicStatsEvaluator(init_vars[0], logger=init_vars[-2], debug_logger=init_vars[-1])
	score = stats_evaluator.evaluate(init_vars[4])

	assert score is not None

def test_basic_stats_unequal(init_vars):
	stats_evaluator = BasicStatsEvaluator(init_vars[0], logger=init_vars[-2], debug_logger=init_vars[-1])
	score = stats_evaluator.evaluate(init_vars[6])

	assert score is not None

def test_basic_stats_zeros(init_vars):
	stats_evaluator = BasicStatsEvaluator(init_vars[2], logger=init_vars[-2], debug_logger=init_vars[-1])
	score = stats_evaluator.evaluate(init_vars[4])

	assert score is not None

def test_basic_stats_cat(init_vars):
	stats_evaluator = BasicStatsEvaluator(init_vars[1], logger=init_vars[-2], debug_logger=init_vars[-1])
	score = stats_evaluator.evaluate(init_vars[5], cat_cols=[2, 3])

	assert score is not None

def test_pmse_equal(init_vars):
	pmse_evaluator = pMSEEvaluator(init_vars[0], num_features=10, logger=init_vars[-2], debug_logger=init_vars[-1])
	score = pmse_evaluator.evaluate(init_vars[4])

	assert score is not None

def test_pmse_unequal(init_vars):
	pmse_evaluator = pMSEEvaluator(init_vars[0], num_features=10, logger=init_vars[-2], debug_logger=init_vars[-1])
	score = pmse_evaluator.evaluate(init_vars[6])

	assert score is not None

def test_pmse_zeros(init_vars):
	pmse_evaluator = pMSEEvaluator(init_vars[2], num_features=10, logger=init_vars[-2], debug_logger=init_vars[-1])
	score = pmse_evaluator.evaluate(init_vars[4])

	assert score is not None

def test_ml_eff_equal(init_vars):
	ml_eff_evaluator = MLEfficacyEvaluator(init_vars[0], num_features=10, logger=init_vars[-2], debug_logger=init_vars[-1])
	score = ml_eff_evaluator.evaluate(init_vars[4])

	assert score is not None

def test_ml_eff_unequal(init_vars):
	ml_eff_evaluator = MLEfficacyEvaluator(init_vars[0], num_features=10, logger=init_vars[-2], debug_logger=init_vars[-1])
	score = ml_eff_evaluator.evaluate(init_vars[6])

	assert score is not None

def test_ml_eff_zeros(init_vars):
	ml_eff_evaluator = MLEfficacyEvaluator(init_vars[2], num_features=10, logger=init_vars[-2], debug_logger=init_vars[-1])
	score = ml_eff_evaluator.evaluate(init_vars[4])

	assert score is not None

def test_sup_cov_equal(init_vars):
	sup_cov_evaluator = SupportCoverageEvaluator(init_vars[0])
	score = sup_cov_evaluator.evaluate(init_vars[4])

	assert score is not None

def test_sup_cov_unequal(init_vars):
	sup_cov_evaluator = SupportCoverageEvaluator(init_vars[0])
	score = sup_cov_evaluator.evaluate(init_vars[6])

	assert score is not None

def test_sup_cov_zeros(init_vars):
	sup_cov_evaluator = SupportCoverageEvaluator(init_vars[2])
	score = sup_cov_evaluator.evaluate(init_vars[4])

	assert score is not None

def test_ft_dist_equal(init_vars):
	ft_dist_evaluator = FTDistEvaluator(init_vars[0])
	score = ft_dist_evaluator.evaluate(init_vars[4], np.array([1.0]*10))

	assert score is not None

def test_ft_dist_unequal(init_vars):
	ft_dist_evaluator = FTDistEvaluator(init_vars[0])
	score = ft_dist_evaluator.evaluate(init_vars[6], np.array([1.0]*10))

	assert score is not None

def test_ft_dist_zeros(init_vars):
	ft_dist_evaluator = FTDistEvaluator(init_vars[2])
	score = ft_dist_evaluator.evaluate(init_vars[4], np.array([1.0]*10))

	assert score is not None