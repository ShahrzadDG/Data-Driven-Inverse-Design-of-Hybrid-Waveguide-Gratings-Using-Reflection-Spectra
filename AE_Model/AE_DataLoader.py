import pandas as pd
import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split
import cv2 
from torch.utils.data import DataLoader, TensorDataset

class ReflectionDataLoader:
    def __init__(self, start_index, end_index, folder_path, test_size, rnd_numb, batch_size):
        self.start_index = start_index
        self.end_index = end_index
        self.folder_path = folder_path
        self.test_size = test_size
        self.rnd_numb = rnd_numb
        self.batch_size = batch_size

    def load_data(self):
        os.chdir(self.folder_path)
        
        Reflection_list = []
        
        for i in range(self.start_index, self.end_index + 1):
            reflection = pd.read_csv(f"Reflection_TM_{i}.csv", header=None)
            reflection = reflection.iloc[1:, 1:].astype(float)

            Reflection_list.append(reflection.values)

        Reflection_data = np.stack(Reflection_list)

        filenames = [f"Reflection_TM_{i}.csv" for i in range(self.start_index, self.end_index + 1)]
        X_train, X_test , filenames_train, filenames_test= train_test_split(Reflection_data,filenames,test_size=self.test_size,random_state=self.rnd_numb)

        with open("Test_filenames-autoencoder-noise.txt", "w") as f:
            for name in filenames_test:
                f.write(name + "\n")
        
        # print(X_train.shape)
        # print(X_test.shape)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)    

        train_dataset = TensorDataset(X_train_tensor, X_train_tensor) 
        test_dataset = TensorDataset(X_test_tensor, X_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)

        return train_loader, test_loader


