import os
import numpy as np
import pandas as pd
import trimesh
import h5py
from tqdm import tqdm

def process_and_save_point_clouds(csv_file, root_dir, output_file, num_points=1024):
    """
    Processes .OFF files into point clouds and saves them in HDF5 format.
    
    Args:
        csv_file (str): Path to the CSV file with annotations.
        root_dir (str): Directory containing the .OFF files.
        output_file (str): Path to save the processed HDF5 file.
        num_points (int): Number of points to sample from each model.
    """
    # Read the CSV file
    data_df = pd.read_csv(csv_file)
 
    point_clouds = []
    labels = []
    splits = []  # To store train/test split information
    
    # Create class mapping
    classes = sorted(data_df['class'].unique())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    print(f"Found {len(classes)} classes: {classes}")
    print(f"Processing {len(data_df)} files...")
    
    # Process each file with progress bar
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
        try:
            object_path = os.path.join(root_dir, row['object_path'])
            
            # Load the .OFF file using trimesh
            mesh = trimesh.load(object_path)
            
            # Sample points from the mesh surface
            points = mesh.sample(num_points)
            
            # Center the point cloud
            centroid = np.mean(points, axis=0)
            points = points - centroid
            
            # Scale to unit sphere
            furthest_distance = np.max(np.sqrt(np.sum(points**2, axis=1)))
            points = points / furthest_distance
            
            # Append to lists
            point_clouds.append(points)
            labels.append(class_to_idx[row['class']])
            splits.append(0 if row['split'] == 'train' else 1)  # 0 for train, 1 for test
            
        except Exception as e:
            print(f"Error processing {row['object_path']}: {e}")
    
    # Convert to numpy arrays
    point_clouds = np.array(point_clouds, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    splits = np.array(splits, dtype=np.int64)
    
    print(f"Processed point clouds shape: {point_clouds.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Save to HDF5 file
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('point_clouds', data=point_clouds, compression='gzip')
        f.create_dataset('labels', data=labels)
        f.create_dataset('splits', data=splits)  # Store train/test split information
        
        # Store class names as attributes
        for i, cls_name in enumerate(classes):
            f.attrs[f'class_{i}'] = cls_name
    
    print(f"Processed data saved to {output_file}")
    
    # Print dataset statistics
    train_count = np.sum(splits == 0)
    test_count = np.sum(splits == 1)
    print(f"Training samples: {train_count}")
    print(f"Testing samples: {test_count}")
    
    # Print class distribution
    for i, cls_name in enumerate(classes):
        class_count = np.sum(labels == i)
        print(f"Class {cls_name}: {class_count} samples")

# Example usage
csv_file = 'E:\Deep_Learning\CADDY\Dataset\metadata_modelnet10.csv'  # Path to the CSV file
root_dir = 'E:\Deep_Learning\CADDY\Dataset\ModelNet10'  # Root directory containing the dataset
output_file = 'processed_data/modelnet10_point_clouds.h5'  # Output HDF5 file

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Process and save the data
process_and_save_point_clouds(csv_file, root_dir, output_file)
