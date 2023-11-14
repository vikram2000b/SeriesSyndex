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

### Testing (Unit Tests)
We use ```pytest``` to run the unit test. You will have to install pytest using ```pip install pytest``` and then run this command to execute all unit test.
```
pytest
```
