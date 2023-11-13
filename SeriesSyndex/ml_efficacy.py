from torch import nn
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split


from SeriesSyndex.models import LSTMRegressor
from SeriesSyndex.data_utils import MLEfficacyDataset

class MLEfficacyEvaluator:
    def __init__(self, real_dataset, num_feature,  lstm_hidden_size = 64, num_layers = 4,
                 num_loader_workers = 1, epochs = 20, lr = 0.01, batch_size = 128, target_feature = 0):
        '''
        Constructor for ML Efficacy Evaluator. 
        Args:
            real_dataset (torch Dataset): The real dataset for comparison.
            num_feature (int): Number of temporal/series features in the data
            lstm_hidden_size (int): Hidden Size for LSTM Model, which will be used calculate ML Efficacy
            num_layers (int): Number of layers for LSTM Model.
            num_loader_workers (int): Number of workers to use for data loaders
            epochs (int): Number of epochs for model training
            lr (float): Initial Learning rate for the model
            batch_size (int): BBatch size for model training
            target_feature (int): Index of target which should be used to Measure ML efficacy.
        '''
        self.real_dataset = real_dataset
        self.num_workers = num_loader_workers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        self.num_feature = num_feature
        self.target_feature = target_feature
    
    def evaluate(self, synthetic_dataset):
        '''
        Evaluate the ML Efficacy score of a synthetic dataset.
        Args:
            synthetic_dataset (torch Dataset or numpy array): The Synthetic dataset which needs to be evaluated.
        
        Returns:
            Score (int): ML Efficacy score of the model.
        '''
        real_model = LSTMRegressor(input_size=self.num_feature, 
                                    hidden_size=self.lstm_hidden_size,
                                    num_layers=self.num_layers)
        
        real_total_size = len(self.real_dataset)
        real_train_size = int(0.8 * real_total_size)  
        real_val_size = int(0.1 * real_total_size)   
        real_test_size = real_total_size - real_train_size - real_val_size 

        # convert the dataset to have labels
        mle_real_dataset = MLEfficacyDataset(self.real_dataset, target_feature=self.target_feature)

        # create train, val, and test datasets
        train_dataset, val_dataset, test_dataset = random_split(
            mle_real_dataset, 
            [real_train_size, real_val_size, real_test_size]
        )
        
        # Not specifying num_workers due to issues in Windows Machine
        train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        
        val_data_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        self.train(real_model, train_data_loader, val_data_loader)

        real_eval = self.eval_model(real_model, test_data_loader)


        syn_model = LSTMRegressor(input_size=self.num_feature, 
                                    hidden_size=self.lstm_hidden_size,
                                    num_layers=self.num_layers)
        
        syn_total_size = len(synthetic_dataset)
        syn_train_size = int(0.8 * syn_total_size)  
        syn_val_size = int(0.1 * syn_total_size)   
        syn_test_size = syn_total_size - syn_train_size - syn_val_size 

        # convert the dataset to have labels
        mle_syn_dataset = MLEfficacyDataset(synthetic_dataset, target_feature=self.target_feature)

        # create train, val, and test datasets
        syn_train_dataset, syn_val_dataset, syn_test_dataset = random_split(
            mle_syn_dataset, 
            [syn_train_size, syn_val_size, syn_test_size]
        )
        
        # Not specifying num_workers due to issues in Windows Machine
        syn_train_data_loader = DataLoader(syn_train_dataset, batch_size=self.batch_size)
        
        syn_val_data_loader = DataLoader(syn_val_dataset, batch_size=self.batch_size)
        
        syn_test_data_loader = DataLoader(syn_test_dataset, batch_size=self.batch_size)

        self.train(real_model, syn_train_data_loader, syn_val_data_loader)

        syn_eval = self.eval_model(real_model, test_data_loader)

        if syn_eval['loss'] <= real_eval['loss']:
            return eval
        score = np.clip(self.mape(np.array(real_eval['loss']), np.array(syn_eval['loss'])), 0, 1)
        return score



        
    def train(self, model, train_data_loader, val_data_loader):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1, verbose = True)

        for epoch in range(self.epochs):
            model.train()
            losses = []
            for batch in train_data_loader:
                (static_vars, series_vars), labels = batch
                
                outputs = model(series_vars)

                loss = loss_fn(outputs.squeeze(), labels.float())
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            val_eval = self.eval_model(model, val_data_loader)

            scheduler.step(val_eval['loss'])

            print(f"Training Epoch: {epoch}, Train Loss: {np.mean(losses)}, \
                  Val Loss: {val_eval['loss']}")
            

    @torch.no_grad()
    def eval_model(self, model, test_data_loader):
        model.eval()
        losses = []
        loss_fn = nn.MSELoss()
        predicted_labels = []
        target_labels = []
        for batch in test_data_loader:
            (static_vars, series_vars), labels = batch
                
            outputs = model(series_vars)
            loss = loss_fn(outputs.squeeze(), labels.float())
            losses.append(loss.item())
            target_labels.append(labels.cpu())
            predicted_labels.append(outputs.squeeze().cpu())


        target_labels = torch.concat(target_labels, axis = 0)
        predicted_labels = torch.concat(predicted_labels, axis = 0)


        test_loss = np.mean(losses)
        return {
            "loss": test_loss
        }
    
    def mape (self, vector_a, vector_b):
        return abs(vector_a-vector_b)/abs(vector_a+1e-6)

    