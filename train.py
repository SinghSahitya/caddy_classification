import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import time
from dataset import PointCloudDataset, PointCloudTransforms
from model import PointNet2Classification

def train_pointnet(h5_file, num_classes=10, batch_size=32, epochs=50, learning_rate=0.001, 
                   checkpoint_dir='checkpoints'):
    """
    Train the PointNet++ model
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders with the custom transform
    train_transforms = PointCloudTransforms(jitter_scale=0.01, rotate=True, jitter=True)
    
    train_dataset = PointCloudDataset(h5_file, split='train', transform=train_transforms)
    test_dataset = PointCloudDataset(h5_file, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    # Create model
    model = PointNet2Classification(num_classes=num_classes).to(device)
    
    # Create optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    criterion = nn.NLLLoss()
    
    # Training loop
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Epoch {epoch+1}/{epochs}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Training phase
        train_pbar = tqdm(train_loader, desc=f"Training")
        for points, labels in train_pbar:
            points, labels = points.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(points)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({'Loss': running_loss/len(train_pbar), 
                                   'Acc': 100.*correct/total})
        
        train_acc = 100. * correct / total
        
        # Testing phase
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f"Testing")
            for points, labels in test_pbar:
                points, labels = points.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(points)
                loss = criterion(outputs, labels)
                
                # Statistics
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                test_pbar.set_postfix({'Loss': test_loss/len(test_pbar), 
                                      'Acc': 100.*correct/total})
        
        test_acc = 100. * correct / total
        
        # Print statistics
        print(f"Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {test_acc:.2f}%")
        
        # Save checkpoint if this is the best model so far
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint_path = os.path.join(checkpoint_dir, f'pointnet_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, checkpoint_path)
            print(f"Saved checkpoint with accuracy: {best_acc:.2f}%")
        
        # Step the scheduler
        scheduler.step()
    
    print(f"Best test accuracy: {best_acc:.2f}%")
    return model, best_acc
