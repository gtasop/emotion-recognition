import os
import re
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Read the labels CSV file
        self.labels_df = pd.read_csv(labels_file)

        # Create a list of image paths and their corresponding labels
        self.image_paths = []
        self.labels = []
        self.pattern = r"trial_(\d+)\\channel_([A-Za-z0-9]+)_psd\.png"

        for idx in range(len(self.labels_df)):
            trial_name = self.labels_df.iloc[idx, 0]
            label_row = self.labels_df.iloc[idx, 1:].values.astype(float)

            # Extract parts from the trial name
            parts = re.search(self.pattern, trial_name)
            trial_number = parts.group(1)
            channel_name = parts.group(2)

            img_name = f'channel_{channel_name}_psd.png'
            img_path = os.path.join(self.root_dir, f'trial_{trial_number}', img_name)

            if os.path.exists(img_path):  # Ensure the image path exists
                self.image_paths.append(img_path)
                self.labels.append(label_row)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        labels = self.labels[idx]

        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)

        # Convert labels to a tensor
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return image, labels_tensor
