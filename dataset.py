import torch
import torch.utils.data
import h5py
import numpy as np
import random

class PointCloudTransforms:
    def __init__(self, jitter_scale=0.01, rotate=True, jitter=True):
        self.jitter_scale = jitter_scale
        self.rotate = rotate
        self.jitter = jitter
        
    def __call__(self, point_cloud):
        if self.rotate:
            # Random rotation around z-axis
            theta = random.random() * 2 * np.pi
            rotation_matrix = torch.tensor([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ], dtype=torch.float32)
            point_cloud = torch.matmul(point_cloud, rotation_matrix)
        
        if self.jitter:
            # Add random noise
            noise = torch.randn_like(point_cloud) * self.jitter_scale
            point_cloud = point_cloud + noise
        
        return point_cloud

class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, h5_file, split='train', transform=None):
        """
        Args:
            h5_file (str): Path to the HDF5 file containing point clouds
            split (str): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.h5_file = h5_file
        self.split = 0 if split == 'train' else 1
        self.transform = transform
        
        # Load the class names
        with h5py.File(h5_file, 'r') as f:
            # Get number of classes
            self.num_classes = 0
            while f'class_{self.num_classes}' in f.attrs:
                self.num_classes += 1
            
            # Get class names, handling both string and bytes formats
            self.classes = []
            for i in range(self.num_classes):
                attr = f.attrs[f'class_{i}']
                if isinstance(attr, bytes):
                    self.classes.append(attr.decode('utf-8'))
                else:
                    self.classes.append(attr)
            
            # Get indices for the specified split
            self.indices = np.where(f['splits'][()] == self.split)[0]

    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            index = self.indices[idx]
            point_cloud = f['point_clouds'][index]
            label = f['labels'][index]
        
        # Convert to PyTorch tensors
        point_cloud = torch.from_numpy(point_cloud.astype(np.float32))
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            point_cloud = self.transform(point_cloud)
            
        return point_cloud, label
