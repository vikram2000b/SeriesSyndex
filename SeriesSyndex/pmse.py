
import torch.nn.init as init
from torch import nn
from torch.utils.data import DataLoader, random_split

from SeriesSyndex.data_utils import pMSEDataset
from SeriesSyndex.models import LSTMClassifier


class pMSEEvaluator:
    def __init__(self, real_dataset, num_features, lstm_hidden_size = 64, num_layers = 2,
                 num_loader_workers = 2):
        self.real_dataset = real_dataset
        self.num_workers = num_loader_workers
        self.lstm_classifier = LSTMClassifier(input_size=num_features, 
                                              hidden_size=lstm_hidden_size,
                                              num_layers=num_layers
                                              )
        
    def reset_weights(self):
        for layer in self.model.children():
            if isinstance(layer, nn.LSTM):
                # Reset LSTM weights
                for name, param in layer.named_parameters():
                    if 'weight' in name:
                        init.xavier_normal_(param.data)
                    elif 'bias' in name:
                        init.zeros_(param.data)
            elif isinstance(layer, nn.Linear):
                # Reset Dense (Linear) layer weights
                init.xavier_normal_(layer.weight.data)
                init.zeros_(layer.bias.data)
        

    def evaluate(self, synthetic_dataset, balance_dataset = True):
        '''
        Function to evaluate the pMSE score of the synthetic dataset. It will train a
        classifier to distinguish between real and synthetic data and then use the
        performance of classifier to calculate the score.

        Args:
            synthetic_dataset: The synthetic which needs to be evaluated
            balance_datase (bool): Whether to balance the datasets to have equal samples for training of classifier.
        '''
        # reset the weights of the model
        self.reset_weights()

        # create dataset for pMSE training.  
        dataset = pMSEDataset(real_dataset=self.real_dataset, 
                                    synthetic_dataset=synthetic_dataset)
        
        total_size = len(dataset)
        train_size = int(0.8 * total_size)  
        val_size = int(0.1 * total_size)   
        test_size = total_size - train_size - val_size 

        # create train, val, and test datasets
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, 
            [train_size, val_size, test_size]
        )
        
        train_data_loader = DataLoader(train_dataset, num_workers=self.num_workers, 
                                batch_size=self.batch_size)
        
        val_data_loader = DataLoader(val_dataset, num_workers=self.num_workers, 
                                batch_size=self.batch_size)
        
        test_data_loader = DataLoader(test_dataset, num_workers=self.num_workers, 
                                batch_size=self.batch_size)


        self.train_model(train_data_loader, val_data_loader)

        test_scores = self.eval_model(test_data_loader)

        # TODO

    def train_mode(self, train_data_loader, val_data_loader):
        # TODO
        pass

    def eval_model(self, test_data_loader):
        # TODO
        pass
