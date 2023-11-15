# SeriesSyndex

This framework is a tool for measuring the goodness of synthetic time-series data. 

## How to run?

### Installation
To install the package, clone the repo and type this command after after opening the project directory.
```
pip install .
```

### Evaluate synthetic data
The evaluator takes datasets in form of pytorch Datasets or numpy arrays. It is advised to use pytorch Datasets for large arrays so there is no need to load the whole dataset in memory.
Here is an example code to evaluate a synthetic data.
```
from SeriesSyndex import Evaluator
from torch.utils.data import Dataset, random_split, DataLoader

series_x = np.load("./FCC_MBA_Dataset/data_train.npz")['data_feature']
static_x = np.load("./FCC_MBA_Dataset/data_train.npz")['data_attribute']

class CustomDataset(Dataset):
    def __init__(self, series_x, static_x):
        super().__init__()
        
        assert series_x.shape[0] == static_x.shape[0]
        self.series_x = series_x
        self.static_x = static_x
        
    def __getitem__(self, index):
        return self.static_x[index], self.series_x[index]
    
    def __len__(self):
        return len(self.series_x)

dataset = CustomDataset(series_x, static_x)

total_size = len(dataset)
real_size = int(0.5 * total_size) 
syn_size = total_size - real_size 

real_dataset, syn_dataset = random_split(
    dataset, 
    [real_size, syn_size]
)

evaluator = Evaluator(real_dataset=real_dataset, num_features = series_x.shape[-1])

# cat_cols is the list of the indexes of categorical columns, default value = []
print(evaluator.evaluate(syn_dataset, cat_cols=[]))
```

Here, we have used a subset of data as the synthetic data for evaluation. Also, the dataset is expected to return two variables, one is representing the static variables and other time-series variables. 
The real dataset is passed to the evaluator when it is initialized and synthetic data is passed while calling ```evaluate``` function.

## Components of the Score
The evaluator aggregates 5 different metrics, which measure different aspects of the synthetic data. The metrics/evaluators are as follows:
1. PMSE Evaluator (Adversary Success Rate Evaluator) - **Production Ready**
2. Basic Statics Evaluator
3. Support Coverage Evaluator
4. ML Efficacy
5. Fourier Transform Distance Evaluator - Not Scalable At the moment

### PMSE Evaluator
This score employs an adversary, which is trained to distinguish between the real data and synthetic data. We then measure the accuracy of the adversary to evaluate the goodness of synthetic data. If an adversary can easily distinguish between real data and synthetic data, the PMSE score will be low i.e. closer to 0. And if the adversary can not distinguish between real and synthetic data, the PMSE Score will be high i.e. closer 1. 
This evaluator trains a neural network to differentiate between real and synthetic data. The synthetic data is assigned a label 1, and real data is assigned a label 0. The training set is created by combining both real and synthetic data with given assigned labels. The accuracy on the test samples is used to calculate the score. The score is calculated as below:

$pmse\_score = 2*(1 - max(accuracy, 0.5))$

Here, an accuracy of 0.5 or less means that model has failed to learn to distinguish between real and synthetic samples, leading to score of 1.
We have provided options for different models as adversary. Following are the available model:
1. LSTM: It is an LSTM based model with a classification layer at the end. It uses several layers of LSTM and then Dense layer to predict the classification output.
2. TCN: It is an Temporal CNN based implementation of the classifier. It uses several layers of Temporal CNNs and then Dense layer to make the predictions. The TCN was provided to be used for larger time-series datasets, where LSTM might be too slow due to it's recurrent and non-scalable nature. **The TCN is a more scalable model than the LSTM**, so it is advised to use this model with larger datasets.

Here is an example code on how to separately use the PMSE Evaluator.

```
from SeriesSyndex.pmse import pMSEEvaluator
from SeriesSyndex.logger import setup_logger

logger = setup_logger("run.log", level = logging.INFO)
debug_logger = setup_logger("debug.log", level = logging.DEBUG)

real_dataset = ... # A pytorch Dataset.
num_features = ... # Number of features in the time series component of the dataset
pmse_evaluator = pMSEEvaluator(real_dataset, num_features=self.num_features, logger=logger, 
                                                debug_logger=debug_logger, batch_size=256, max_batches = None,
                                                device = 'cuda', model_type= 'TCN')

synthetic_dataset = ... # the synthetic dataset for measurement
score = pmse_evaluator.evaluate(synthetic_data)

print(f"PMSE Score of the synthetic data: {score}")
```

### Scalability plan for PMSE
The PMSE is currently running on a single machine and only using Single GPU. It can be further scaled with Distributed Training.
The ```pmse_distributed.py``` has the code for distributed training. The code is not yet complete and has not been tested yet due to resource constraints.

### Basic Stats Evaluator
This score measure the closeness of simple statistics like mean, standard deviation, and correlation of all the features of synthetic data to that of real data. 

A dataset of shape (num_samples, num_time_steps, num_features) is converted into tabular data of shape (num_samples*num_time_steps, num_features) and then mean of all the features, standard deviation of all the features, and correlation within all the features is calculated. If categorical columns exist, Theil's U is used as a substitute for correlation among them.

$basic\_stats\_score = 1 - \frac{1}{3}(MAPE(real\_mean, syn\_mean)+MAPE(real\_std, syn\_std)+MAPE(real\_corr, syn\_corr))$

Log of the values in correlation matrix is used to avoid instability due to extremely small correlation values.

Here is an example code on how to separately use the Basic Stats Evaluator.

```
from SeriesSyndex.basic_stats import BasicStatsEvaluator
from SeriesSyndex.logger import setup_logger

logger = setup_logger("run.log", level = logging.INFO)
debug_logger = setup_logger("debug.log", level = logging.DEBUG)

real_dataset = ... # A pytorch Dataset
stats_evaluator = BasicStatsEvaluator(real_dataset, logger=logger, debug_logger=debug_logger)

synthetic_dataset = ... # the synthetic dataset for measurement
score = stats_evaluator.evaluate(synthetic_data)

print(f"Basic Stats Score of the synthetic data: {score}")
```

### Support Coverage Evaluator
This score evaluates the distribution of values in different ranges in synthetic data when compared to real data.

A dataset of shape (num_samples, num_time_steps, num_features) is converted into tabular data of shape (num_samples*num_time_steps, num_features) and num_bins (default: 20) number of bins are created dividing the range between minimum and maximum value of each feature.

If the size of real and synthetic data is same,

$sup\_cov\_score = \frac{1}{num\_features}\frac{1}{num\_bins}\sum_{i}\frac{num\_samples\_syn\_bin\_i}{num\_samples\_real\_bin\_i}$


Here is an example code on how to separately use the Support Coverage Evaluator.

```
from SeriesSyndex.support_coverage import SupportCoverageEvaluator
from SeriesSyndex.logger import setup_logger

logger = setup_logger("run.log", level = logging.INFO)
debug_logger = setup_logger("debug.log", level = logging.DEBUG)

real_dataset = ... # A pytorch Dataset.
num_features = ... # Number of features in the time series component of the dataset
sup_cov_evaluator = SupportCoverageEvaluator(real_dataset, logger=logger, 
                                                              debug_logger=debug_logger)

synthetic_dataset = ... # the synthetic dataset for measurement
score = sup_cov_evaluator.evaluate(synthetic_data)

print(f"Support Coverage Score of the synthetic data: {score}")
```

### Machine Learning Efficacy Evaluator
This score measures how useful the synthetic data is for machine learning purposes. Real train set is used to train a ML model (we have implemented LSTM and TCN, same as in PMSE Evaluator) and its loss on real test set is recorded. We then train the same ML model on synthetic data, and test how close the performance on real test set gets to that of real train set.

$ml\_efficay\_score = 1 - MAPE(real\_loss, syn\_loss)$

Here is an example code on how to separately use the ML Efficacy Evaluator.

```
from SeriesSyndex.ml_efficacy import MLEfficacyEvaluator
from SeriesSyndex.logger import setup_logger

logger = setup_logger("run.log", level = logging.INFO)
debug_logger = setup_logger("debug.log", level = logging.DEBUG)

real_dataset = ... # A pytorch Dataset.
num_features = ... # Number of features in the time series component of the dataset
ml_eff_evaluator = MLEfficacyEvaluator(real_dataset, num_features=self.num_features, 
                                                        logger=logger, debug_logger=debug_logger,
                                                        device = 'cuda', model_type='TCN')

synthetic_dataset = ... # the synthetic dataset for measurement
score = sup_cov_evaluator.evaluate(synthetic_data)

print(f"ML Efficay Score of the synthetic data: {score}")
```

### Fourier Transform Distance Evaluator
This score measures the similarity between the Fourier Transforms of real and synthetic data.

A dataset of shape (num_samples, num_time_steps, num_features) on Fourier Transform yeilds a array of the same shape but with complex numbers. Top n (default:50) of this transformed data aree chosen and this is converted into tabular data of shape (num_samples*n_components, num_features). Real and imaginary part are treated as x, y coordinates and Wasserstein distance is measured between real transformed data and synthetic transformed data. To get the score, this distance is divided by wasserstein distance between subsets of real data.

$ft\_dist\_score = \frac{1}{num\_features}\sum_i\frac{wass\_dist(real, syn)}{wass\_dist(real, real)}$

```
from SeriesSyndex.ft_dist import FTDistEvaluator
from SeriesSyndex.logger import setup_logger

logger = setup_logger("run.log", level = logging.INFO)
debug_logger = setup_logger("debug.log", level = logging.DEBUG)

real_dataset = ... # A pytorch Dataset.
num_features = ... # Number of features in the time series component of the dataset
ft_dist_evaluator = FTDistEvaluator(real_dataset, logger=logger, debug_logger=debug_logger, max_batches = 50)

synthetic_dataset = ... # the synthetic dataset for measurement
score = ft_dist_evaluator.evaluate(synthetic_data)

print(f"FT Dist Score of the synthetic data: {score}")
```

### Testing (Unit Tests)
We use ```pytest``` to run the unit test. You will have to install pytest using ```pip install pytest``` and then run this command to execute all unit test.
```
pytest
```
