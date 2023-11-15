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

## Components of the Score
The evaluator aggregates 5 different metrics, which measure different aspects of the synthetic data. The metrics/evaluators are as follows:
1. PMSE Evaluator (Adversary Success Rate Evaluator) - **Production Ready**
2. Basic Statics Evaluator
3. Support Coverage Evaluator
4. ML Efficacy
5. Fourier Transform Distance Evaluator - Not Scalable At the moment

### PMSE Evaluator
This score employs an adversary, which is trained to distinguish between the real data and synthetic data. We then measure the accuracy of the adversary to evaluate the goodness of synthetic data. If an adversary can easily distinguish between real data and synthetic data, the PMSE score will be low i.e. closer to 0. And if the adversary can not distinguish between real and synthetic data, the PMSE Score will be high i.e. closer 1. 
This evaluator trains a neural network to differentiate between real and synthetic data. The synthetic data is assigned a label 1, and real data is assigned a label 0. The training set is created by combining both real and synthetic data with give assigned labels. The accuracy on the test samples is used to calculate the score. The score is calculated as below:
```
pmse_score = 2*(1 - max(accuracy, 0.5))
```
Here, an accuracy of 0.5 or less means that model has failed to learn to distinguish between real and synthetic samples, leading to score of 1.
We have provided options for different models as adversary. Following are the available model:
1. LSTM: It is an LSTM based model with a classification layer at the end. It uses several layers of LSTM and then Dense layer to predict the classification output.
2. TCN: It is an Temporal CNN based implementation of the classifier. It uses several layers of Temporal CNNs and then Dense layer to make the predictions. The TCN was provided to be used for larger time-series datasets, where LSTM might be too slow due to it's recurrent and non-scalable nature. **The TCN is a scalable model than the LSTM**, so it is advised to use this model with larger datasets.

Here is an example code on how to separately use the PMSE Evaluator.

```
from SeriesSyndex.pmse import pMSEEvaluator
from SeriesSyndex.logger import setup_logger

logger = setup_logger("run.log", level = logging.INFO)
debug_logger = setup_logger("debug.log", level = logging.DEBUG)

real_dataset = ... # A pytorch Dataset.
num_features = ... # Number of features in the time series component of the dataset
self.pmse_evaluator = pMSEEvaluator(real_dataset, num_features=self.num_features, logger=logger, 
                                                debug_logger=debug_logger, batch_size=256, max_batches = None,
                                                device = 'cuda', model_type= 'TCN')

synthetic_dataset = ... # the synthetic dataset for measurement
score = self.pmse_evaluator.evaluate(synthetic_data)

print(f"PMSE Score of the synthetic data: {score}")
```
###




Here, we have used a subset of data as the synthetic data for evaluation. Also, the dataset is expected to return two variables, one is representing the static variables and other time-series variables. 
The real dataset is passed to the evaluator when it is initialized and synthetic data is passed while calling ```evaluate``` function.

### Testing (Unit Tests)
We use ```pytest``` to run the unit test. You will have to install pytest using ```pip install pytest``` and then run this command to execute all unit test.
```
pytest
```
