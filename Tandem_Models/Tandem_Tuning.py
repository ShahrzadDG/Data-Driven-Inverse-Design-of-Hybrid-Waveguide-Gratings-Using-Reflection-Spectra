# Tuning the hyperparameters - optuna
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
from torch.optim.lr_scheduler import StepLR
from optuna.integration import PyTorchLightningPruningCallback
import random

random.seed(16)
np.random.seed(16)
torch.manual_seed(16)
torch.cuda.manual_seed(16)
torch.cuda.manual_seed_all(16)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
import sys
sys.path.append('/path to codes/')
from Reflection_Structure_DataLoader import ReflStrDataLoader
from Models import Autoencoder, StructureToLatent

class LatenttoStructureOptuna(nn.Module):
    def __init__(self, input_dim=128, output_dim=3, n_layers=4, n_units_per_layer=None, activation=nn.ReLU, dropout_rate=0.0):
        super(LatenttoStructureOptuna, self).__init__()
        
        layers = []
        input_dim = 128
        output_dim = 3

        layers.append(nn.Linear(input_dim, n_units_per_layer[0]))
        layers.append(activation)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
            
        for i in range(1, n_layers):
            layers.append(nn.Linear(n_units_per_layer[i-1], n_units_per_layer[i]))
            layers.append(activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(n_units_per_layer[-1], output_dim))

        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)    
    
class TandemNetwork(nn.Module):
    def __init__(self, forward_model, inverse_model):
        super(TandemNetwork, self).__init__()
        self.forward_model = forward_model
        self.inverse_model = inverse_model

    def forward(self, y):
        pred_structure = self.inverse_model(y)
        latent_pred = self.forward_model(pred_structure)
        return latent_pred

    def pred(self, y):
        pred_structure = self.inverse_model(y)
        return pred_structure

def load_structure_to_latent_model(model_path, input_dim=3, output_dim=128, device='cpu'):
    model = StructureToLatent(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model
    
def objective(trial):
    n_layers = trial.suggest_int('n_layers', 3, 9)
    n_units_per_layer = [trial.suggest_int(f'n_units_l{i}', 32, 512, step=32) for i in range(n_layers)]
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    activation_name = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])
    if activation_name == 'ReLU':
        activation = nn.ReLU()
    elif activation_name == 'LeakyReLU':
        negative_slope = trial.suggest_float('leaky_relu_negative_slope', 1e-3, 0.3)
        activation = nn.LeakyReLU(negative_slope=negative_slope)
    else:
        activation = nn.Tanh()
        
    forward_model = load_structure_to_latent_model("/path to forward model/Forward_model_Strtolatent_optuna_NoisyAE.pth", device=device)
    forward_model.eval()

    inverse_model = LatenttoStructureOptuna(input_dim=128, output_dim=3, n_layers=n_layers, n_units_per_layer=n_units_per_layer, activation=activation, dropout_rate=dropout_rate).to(device)
    
    model = TandemNetwork(forward_model=forward_model, inverse_model=inverse_model).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.inverse_model.parameters(), lr=trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True))

    decay_step = trial.suggest_int('decay_step', 1000, 5000)
    decay_factor = trial.suggest_float('decay_factor', 0.1, 0.9)
    scheduler = StepLR(optimizer, step_size=decay_step, gamma=decay_factor)

    num_epochs = 30
    patience = 100
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(num_epochs):
        model.inverse_model.train()
        model.forward_model.eval()  

        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            latent_pred = model(inputs)
            loss = criterion(latent_pred, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                latent_pred = model(inputs)
                loss = criterion(latent_pred, inputs)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            break

        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss


if __name__ == "__main__":
    start_index = 0
    end_index = 14999
    folder_path = "/path to data/"
    test_size = 0.2
    rnd_numb = 42
    batch_size = 128
    num_epochs = 50
    patience = 1000

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_loader = ReflStrDataLoader(start_index, end_index, folder_path, test_size, rnd_numb, batch_size, inverse=True, filename="Test_filenames-tandem_tuning.txt")
    train_loader, val_loader = data_loader.load_data()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=200)

    print("Best trial:")
    trial = study.best_trial

    print(f"Value:{trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"{key}:{value}")

