
import torch.nn.init as init
from torch import nn
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score


from SeriesSyndex.data_utils import pMSEDataset
from SeriesSyndex.models import LSTMClassifier, TCNClassifier


class pMSEEvaluator:
    def __init__(self, real_dataset, num_features, logger, debug_logger, lstm_hidden_size = 64, num_layers = 4,
                 num_loader_workers = 1, epochs = 20, lr = 0.01, batch_size = 128,
                    num_channels = 64, kernel_size = 3, model_type = 'TCN', max_batches = None, device = 'cpu'):
        '''
        Constructor ofr pMSE Evaluator.
        Args:
            real_dataset 
        '''
        self.debug_logger = debug_logger
        self.debug_logger.info("Initiating the ML Efficacy Evaluator Class.")
        self.logger = logger
        self.real_dataset = real_dataset
        self.num_workers = num_loader_workers
        self.device = device
        if model_type == 'TCN':
            self.model = TCNClassifier(input_size=num_features, num_channels=num_channels,
                                      kernel_size = kernel_size, num_layers = num_layers).to(device)
        elif model_type == 'LSTM':
            self.model = LSTMClassifier(input_size=num_features, 
                                                hidden_size=lstm_hidden_size,
                                                num_layers=num_layers
                                                ).to(device)
        else:
            self.logger.info(f"The model type {self.model_type} is not supported.")
            self.debug_logger.debug(f"The model type {self.model_type} is not supported.")
            raise Exception(f"The model type {self.model_type} is not supported.")
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.max_batches = max_batches
        
    def reset_weights(self):
        '''Function to re-initialize the weights of the model.'''
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

        if balance_dataset:
            # TODO
            pass

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
        
        # Not specifying num_workers due to issues in Windows Machine
        train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        
        val_data_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        self.debug_logger.info("Training the classifier.")
        self.train_model(train_data_loader, val_data_loader)

        self.debug_logger.info("Evaluating the classifier.")
        test_scores = self.eval_model(test_data_loader)

        test_acc = test_scores['accuracy']

        return 2*(1 - max(test_acc, 0.5))

    def train_model(self, train_data_loader, val_data_loader):
        '''Train the model.
        Args:
            train_data_loader (torch DataLoader): DataLoader for train set.
            val_data_loader (torch DataLoader): Dataloader for Validation Set
        '''

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1, verbose = False)

        for epoch in range(self.epochs):
            self.model.train()
            losses = []
            num_batches_processed = 0
            for batch in train_data_loader:
                (static_vars, series_vars), labels = batch
                outputs = self.model(series_vars.float().to(self.device))

                loss = loss_fn(outputs.squeeze().to(self.device), labels.float().to(self.device))
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num_batches_processed += 1
                if self.max_batches and (num_batches_processed >= self.max_batches):
                    break
            val_eval = self.eval_model(val_data_loader)

            scheduler.step(val_eval['loss'])

            self.debug_logger.debug(f"Training Epoch: {epoch}, Train Loss: {np.mean(losses)}, \
                  Val Loss: {val_eval['loss']}, Val Acc: {val_eval['accuracy']}")
    
    @torch.no_grad()
    def eval_model(self, test_data_loader):
        self.model.eval()
        losses = []
        loss_fn = nn.BCELoss()
        predicted_labels = []
        target_labels = []
        predicted_probs = []
        for batch in test_data_loader:
            (static_vars, series_vars), labels = batch
                
            outputs = self.model(series_vars.float().to(self.device))
            loss = loss_fn(outputs.squeeze().to(self.device), labels.float().to(self.device))
            losses.append(loss.item())
            target_labels.append(labels.cpu())
            predicted_labels.append((outputs.squeeze() > 0.5).detach().float().cpu())
            predicted_probs.append(outputs.squeeze().cpu().numpy())

        target_labels = torch.concat(target_labels, axis = 0)
        predicted_labels = torch.concat(predicted_labels, axis = 0)
        predicted_probs = np.concatenate(predicted_probs, axis=0)

        accuracy = torch.sum(predicted_labels == target_labels) / target_labels.size(0)
        auc = roc_auc_score(target_labels.cpu().numpy(), predicted_probs)
        test_loss = np.mean(losses)
        return {
            "loss": test_loss,
            "accuracy": accuracy,
            "auc": auc
        }

