from torch import nn
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split


from SeriesSyndex.models import LSTMRegressor, TCNRegressor
from SeriesSyndex.data_utils import MLEfficacyDataset

class MLEfficacyEvaluator:
    def __init__(self, real_dataset, num_features, logger, debug_logger, lstm_hidden_size = 64, num_layers = 4,
                 num_loader_workers = 1, epochs = 20, lr = 0.01, batch_size = 128, target_feature = 0,
                 num_channels = 64, kernel_size = 3, model_type = 'TCN', max_batches = None):
        '''
        Constructor for ML Efficacy Evaluator. 
        Args:
            real_dataset (torch Dataset): The real dataset for comparison.
            num_features (int): Number of temporal/series features in the data
            lstm_hidden_size (int): Hidden Size for LSTM Model, which will be used calculate ML Efficacy
            num_layers (int): Number of layers for LSTM Model.
            num_loader_workers (int): Number of workers to use for data loaders
            epochs (int): Number of epochs for model training
            lr (float): Initial Learning rate for the model
            batch_size (int): BBatch size for model training
            target_feature (int): Index of target which should be used to Measure ML efficacy.
            num_channels (int): Number of channels for the TCN model.
            kernel_size (int): Kernel Size for the TCN Model
            model_type (string): Model type to use for evaluation. Options - TCN, LSTM. 
        '''
        # TODO: Wrap the model parameters into a dictionary and take that in the argument.
        self.debug_logger = debug_logger
        self.debug_logger.info("Initiating the ML Efficacy Evaluator Class.")
        self.real_dataset = real_dataset
        self.num_workers = num_loader_workers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        self.num_features = num_features
        self.target_feature = target_feature
        self.model_type = model_type
        self.logger = logger
        self.debug_logger = debug_logger
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.max_batches = max_batches

        
    
    def evaluate(self, synthetic_dataset):
        '''
        Evaluate the ML Efficacy score of a synthetic dataset.
        Args:
            synthetic_dataset (torch Dataset or numpy array): The Synthetic dataset which needs to be evaluated.
        
        Returns:
            Score (int): ML Efficacy score of the model.
        '''
        self.debug_logger.info("Evaluate function for ML Efficacy Evaluator.")
        
        real_model = None
        self.debug_logger.info("Initiating the prediction model for real data.")
        self.debug_logger.debug(f"Model type: {self.model_type}")
        if self.model_type == 'LSTM':
            real_model = LSTMRegressor(input_size=self.num_features, 
                                        hidden_size=self.lstm_hidden_size,
                                        num_layers=self.num_layers)
        elif self.model_type == 'TCN':
            real_model = TCNRegressor(input_size=self.num_features, num_channels=self.num_channels,
                                      kernel_size = self.kernel_size, num_layers = self.num_layers)
        else:
            self.logger.info(f"The model type {self.model_type} is not supported.")
            self.debug_logger.debug(f"The model type {self.model_type} is not supported.")
            raise Exception(f"The model type {self.model_type} is not supported.")
        self.debug_logger.info("Model initiated.")
        real_total_size = len(self.real_dataset)
        real_train_size = int(0.8 * real_total_size)  
        real_val_size = int(0.1 * real_total_size)   
        real_test_size = real_total_size - real_train_size - real_val_size 
        self.debug_logger.debug(f"Size of Real Dataset - Size: {real_total_size}, Train Split Size: {real_train_size}, \
                          Validation Split Size: {real_val_size}, Test Split Size: {real_test_size}")

        self.debug_logger.debug(f"Creating Custom Dataset for ML Efficacy, to have labels. Target Feature: {self.target_feature}")
        # convert the dataset to have labels
        mle_real_dataset = MLEfficacyDataset(self.real_dataset, target_feature=self.target_feature)

        self.debug_logger.info(f"Creating train, val, and test split for real data.")
        # create train, val, and test datasets
        train_dataset, val_dataset, test_dataset = random_split(
            mle_real_dataset, 
            [real_train_size, real_val_size, real_test_size]
        )
        
        self.debug_logger.info("Creating Data Loader for the splits of real dataset.")
        # Not specifying num_workers due to issues in Windows Machine
        train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        
        val_data_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        self.debug_logger.info("Training the model on real dataset.")
        self.train(real_model, train_data_loader, val_data_loader)
        self.debug_logger.info("Evaluating the trained model on real dataset.")
        real_eval = self.eval_model(real_model, test_data_loader)
        self.debug_logger.debug(f"Eval Result: {real_eval}")

        self.debug_logger.info("Initiating the prediction model for synthetic data.")
        self.debug_logger.debug(f"Model type: {self.model_type}")
        if self.model_type == 'LSTM':
            syn_model = LSTMRegressor(input_size=self.num_features, 
                                        hidden_size=self.lstm_hidden_size,
                                        num_layers=self.num_layers)
        elif self.model_type == 'TCN':
            syn_model = TCNRegressor(input_size=self.num_features, num_channels=self.num_channels,
                                      kernel_size = self.kernel_size, num_layers = self.num_layers)
        else:
            self.logger.info(f"The model type {self.model_type} is not supported.")
            self.debug_logger.info(f"The model type {self.model_type} is not supported.")
            raise Exception(f"The model type {self.model_type} is not supported.")
        self.debug_logger.info("Model initiated")
        
        syn_total_size = len(synthetic_dataset)
        syn_train_size = int(0.8 * syn_total_size)  
        syn_val_size = int(0.1 * syn_total_size)   
        syn_test_size = syn_total_size - syn_train_size - syn_val_size 
        self.debug_logger.debug(f"Size of Synthetic Dataset - Size: {real_total_size}, Train Split Size: {real_train_size}, \
                          Validation Split Size: {real_val_size}, Test Split Size: {real_test_size}")

        self.debug_logger.debug(f"Creating Custom Dataset for ML Efficacy, to have labels. Target Feature: {self.target_feature}")
        # convert the dataset to have labels
        self.debug_logger.debug(f"Creating Custom Dataset for ML Efficacy, to have labels. Target Feature: {self.target_feature}")
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
        self.debug_logger.info("Training the model on synthetic dataset.")
        self.train(syn_model, syn_train_data_loader, syn_val_data_loader)
        self.debug_logger.info("Evaluating the trained model on real dataset.")
        syn_eval = self.eval_model(real_model, test_data_loader)
        self.debug_logger.debug(f"Eval Result: {syn_eval}")
        if syn_eval['loss'] <= real_eval['loss']:
            return 1.
        score = np.clip(self.mape(np.array(real_eval['loss']), np.array(syn_eval['loss'])), 0, 1)
        self.debug_logger.debug(f"ML Efficacy Score: {score}")

        return score



        
    def train(self, model, train_data_loader, val_data_loader):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1, verbose = False)

        for epoch in range(self.epochs):
            model.train()
            losses = []
            num_batches_processed = 0
            for batch in train_data_loader:
                (static_vars, series_vars), labels = batch
                
                outputs = model(series_vars.float())

                loss = loss_fn(outputs.squeeze(), labels.float())
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num_batches_processed += 1
                if self.max_batches and (num_batches_processed >= self.max_batches):
                    break
            val_eval = self.eval_model(model, val_data_loader)

            scheduler.step(val_eval['loss'])

            self.debug_logger.debug(f"Training Epoch: {epoch}, Train Loss: {np.mean(losses)}, \
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
                
            outputs = model(series_vars.float())
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

    