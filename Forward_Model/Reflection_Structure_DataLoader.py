import os
import cv2
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
import sys
sys.path.append('/path to codes/')
from Models import Autoencoder

def load_model_and_reconstruct(input_tensor, model_path='/path to model/autoencoder_model_128_noise.pth', latent_dim=128, device='cpu'):
    model = Autoencoder(latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        reconstructed, latent_vector = model(input_tensor)
    return reconstructed, latent_vector
        
min_values = [10, 0, 280]  
max_values = [100, 1500, 550]  

def min_max_scaler(values, min_values, max_values):
    scaled_values = [(value - min_val) / (max_val - min_val) for value, min_val, max_val in zip(values, min_values, max_values)]
    return scaled_values

def revert_min_max_scaler(normalized_value, min_value, max_value):
    return normalized_value * (max_value - min_value) + min_value
    
class ReflStrDataLoader:
    def __init__(self, start_index, end_index, folder_path, test_size, rnd_numb, batch_size, inverse=False, filename="Test_filenames.txt"):
        self.start_index=start_index
        self.end_index=end_index
        self.folder_path = folder_path
        self.test_size = test_size
        self.rnd_numb = rnd_numb
        self.batch_size = batch_size
        self.inverse = inverse
        self.filename = filename
        
    def load_data(self):
        os.chdir(self.folder_path)
        Structure_list = []
        Reflection_list = []
        file_names = []
        for i in range(self.start_index, self.end_index + 1):
            structure = pd.read_csv("Structure_" + str(i) + ".csv", header=0)
            normalized_structure = structure.apply(lambda row: min_max_scaler(row, min_values, max_values), axis=1)
            Structure_list.append(normalized_structure)
            file_names = np.append(file_names, "Reflection_TM_" + str(i) + ".csv")            
            reflection = pd.read_csv(f"Reflection_TM_{i}.csv", header=None)
            reflection = reflection.iloc[1:, 1:].astype(float)
            reflection = reflection.values
            reflection_tensor = torch.tensor(reflection, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  
            reconstructed_output, latent_vector = load_model_and_reconstruct(reflection_tensor, device=device)

            Reflection_list.append(latent_vector.squeeze(0).cpu().numpy())
            
        Structure_data = pd.concat(Structure_list, ignore_index=True)
        # print(Structure_data.shape)
        
        Reflection_data = np.vstack(Reflection_list)
        # print(Reflection_data.shape)

        if self.inverse:
            X_train, X_test, y_train, y_test, train_file_names, test_file_names = train_test_split(Reflection_data, Structure_data, file_names, test_size=self.test_size, random_state=self.rnd_numb)
        else:
            X_train, X_test, y_train, y_test, train_file_names, test_file_names = train_test_split(Structure_data, Reflection_data, file_names, test_size=self.test_size, random_state=self.rnd_numb)

        with open(self.filename, "w") as f:
            for name in test_file_names:
                f.write(name + "\n")
        
        X_train = np.array(list(X_train), dtype=np.float32)
        # print(X_train.shape)
        X_test = np.array(list(X_test), dtype=np.float32)
        y_train = np.array(list(y_train), dtype=np.float32)  
        y_test = np.array(list(y_test), dtype=np.float32)    
        # print(y_train.shape)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

       
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)

        return train_loader, test_loader



class TandemDataLoader:
    def __init__(self, start_index, end_index, folder_path):
        self.start_index = start_index
        self.end_index = end_index
        self.folder_path = folder_path
    
    def load_data(self):
        os.chdir(self.folder_path)
        Structure_list = []
        Reflection_list = []
        file_names = []
        excluded_files = []

        for i in range(self.start_index, self.end_index + 1):
            structure_path = f"Structure_{i}.csv"
            reflection_path = f"Reflection_TM_{i}.csv"
            # print(structure_path)
            structure = pd.read_csv(structure_path, header=0)
            
            normalized_structure = np.array([min_max_scaler(row, min_values, max_values) for _, row in structure.iterrows()], dtype=np.float32)
            Structure_list.append(normalized_structure)
    
            reflection = pd.read_csv(reflection_path, header=None)
            reflection = reflection.iloc[1:, 1:].astype(float)
            reflection = reflection.values
            reflection_tensor = torch.tensor(reflection, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  
            reconstructed_output, latent_vector = load_model_and_reconstruct(reflection_tensor, device=device)
            Reflection_list.append(latent_vector.squeeze(0).cpu().numpy())
            
            file_names.append(f"Reflection_TM_{i}.csv")
            
        Structures = np.vstack(Structure_list)  
        Reflections = np.array(Reflection_list, dtype=np.float32)  
    
        return Structures, Reflections, file_names


class cVAEDataLoader:
    def __init__(self, start_index, end_index, folder_path, filename="filenames.txt"):
        self.start_index = start_index
        self.end_index = end_index
        self.filename = filename
        self.folder_path = folder_path
    
    def load_data(self):
        os.chdir(self.folder_path)
        Structure_list = []
        Reflection_list = []
        file_names = []
        excluded_files = []

        for i in range(self.start_index, self.end_index + 1):
            structure_path = f"Structure_{i}.csv"
            reflection_path = f"Reflection_TM_{i}.csv"
            # print(reflection_path)
            
            structure = pd.read_csv(structure_path, header=0)
            
            normalized_structure = np.array([min_max_scaler(row, min_values, max_values) for _, row in structure.iterrows()], dtype=np.float32)
            Structure_list.append(normalized_structure)
    
            reflection = pd.read_csv(reflection_path, header=None)
            reflection = reflection.iloc[1:, 1:].astype(float)
            Reflection_list.append(reflection.values.astype(np.float32))  
            
            file_names.append(f"Reflection_TM_{i}.csv")
            
        with open(self.filename, "w") as f:
            for fname in excluded_files:
                f.write(fname + "\n")
        Structures = np.vstack(Structure_list)  
        Reflections = np.array(Reflection_list, dtype=np.float32)  
    
        return Structures, Reflections, file_names
        

