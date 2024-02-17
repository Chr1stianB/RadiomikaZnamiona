# python libraries
from PIL import Image

# pytorch libraries
import torch
from torch.utils.data import DataLoader,Dataset


# Define a pytorch dataloader for this dataset
class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['image_path'][index])
        X_y = Image.open(self.df['mask_path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)
            X_y = self.transform(X_y)
            
        return X, X_y, y
    