import torch
import numpy as np
import os
import argparse
import time
from train import train_pointnet


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train PointNet++ on ModelNet10')
    parser.add_argument('--h5_file', type=str, default='utils/processed_data/modelnet10_point_clouds.h5',
                       help='Path to the HDF5 file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', 
                       help='Directory to save checkpoints')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train the model
    model, best_acc = train_pointnet(
        h5_file=args.h5_file,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir
    )
    
    print(f"Training completed with best accuracy: {best_acc:.2f}%")
