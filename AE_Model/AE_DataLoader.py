import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class ReflectionDataset(Dataset):
    def __init__(self, filenames, folder_path):
        self.filenames = filenames
        self.folder_path = folder_path

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        file_path = os.path.join(self.folder_path, filename)

        reflection = pd.read_csv(file_path, header=None)
        reflection = reflection.iloc[1:, 1:].to_numpy(dtype="float32")

        reflection_tensor = torch.tensor(reflection, dtype=torch.float32).unsqueeze(0)

        return reflection_tensor, reflection_tensor


class ReflectionDataLoader:
    def __init__(
        self,
        start_index,
        end_index,
        folder_path,
        test_size,
        rnd_numb,
        batch_size,
        val_size=0.2
    ):
        self.start_index = start_index
        self.end_index = end_index
        self.folder_path = folder_path
        self.test_size = test_size
        self.val_size = val_size
        self.rnd_numb = rnd_numb
        self.batch_size = batch_size

    def load_data(self):

        filenames = [
            f"Reflection_TM_{i}.csv"
            for i in range(self.start_index, self.end_index + 1)
        ]

        train_val_files, test_files = train_test_split(filenames,test_size=self.test_size,random_state=self.rnd_numb,shuffle=True)
        train_files, val_files = train_test_split(train_val_files,test_size=self.val_size, random_state=self.rnd_numb, shuffle=True)


        with open(os.path.join(self.folder_path, "Test_filenames-autoencoder-noise.txt"), "w") as f:
            for name in test_files:
                f.write(name + "\n")

        train_dataset = ReflectionDataset(train_files, self.folder_path)
        val_dataset = ReflectionDataset(val_files, self.folder_path)
        test_dataset = ReflectionDataset(test_files, self.folder_path)

        train_loader = DataLoader(train_dataset,batch_size=self.batch_size, shuffle=True, num_workers=16)
        val_loader = DataLoader(val_dataset,batch_size=self.batch_size,shuffle=False,num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)

        return train_loader, val_loader, test_loader
