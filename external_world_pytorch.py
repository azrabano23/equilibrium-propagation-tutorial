import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import os

class External_World:
    """PyTorch implementation of the external world (data loading)"""
    
    def __init__(self, data_dir='./data'):
        """
        Initialize the external world with MNIST dataset
        
        Args:
            data_dir (str): Directory to store the dataset
        """
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Download and load MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784 dimensions
        ])
        
        # Load training and test datasets
        train_dataset = datasets.MNIST(
            root=data_dir, 
            train=True, 
            download=True, 
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            root=data_dir, 
            train=False, 
            download=True, 
            transform=transform
        )
        
        # Combine all data (training + test) as in original implementation
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=len(train_dataset), 
            shuffle=False
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=len(test_dataset), 
            shuffle=False
        )
        
        # Extract all data
        train_x, train_y = next(iter(train_loader))
        test_x, test_y = next(iter(test_loader))
        
        # Concatenate training and test data
        self.x = torch.cat([train_x, test_x], dim=0)
        self.y = torch.cat([train_y, test_y], dim=0)
        
        self.size_dataset = len(self.x)
        
        print(f"Loaded MNIST dataset: {self.size_dataset} samples")
        print(f"Input shape: {self.x.shape}")
        print(f"Labels shape: {self.y.shape}")
    
    def get_batch(self, batch_indices):
        """
        Get a batch of data
        
        Args:
            batch_indices (torch.Tensor): Indices of the batch
            
        Returns:
            tuple: (x_batch, y_batch)
        """
        return self.x[batch_indices], self.y[batch_indices]
    
    def get_one_hot_labels(self, batch_indices):
        """
        Get one-hot encoded labels for a batch
        
        Args:
            batch_indices (torch.Tensor): Indices of the batch
            
        Returns:
            torch.Tensor: One-hot encoded labels
        """
        y_batch = self.y[batch_indices]
        return F.one_hot(y_batch, num_classes=10).float()
