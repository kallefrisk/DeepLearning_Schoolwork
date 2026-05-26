import os
import pandas as pd
import numpy as np

def load_and_reshape_all_files(data_dir, n_frames=30, n_features=39):
    """Load all CSV files and reshape them"""
    all_X = []
    all_y = []
    
    for csv_file in os.listdir(data_dir):
        if csv_file.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, csv_file))
            y = df["target"].values
            X_flat = df.drop(columns=["target"]).values
            
            # Reshape
            X = X_flat.reshape(-1, n_frames, n_features)
            
            all_X.append(X)
            all_y.append(y)
    
    # Concatenate all samples
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    
    return X_all, y_all


def mirror_pose_data(X):
    """
    Mirror pose data by swapping left and right body parts.
    
    Args:
        X: numpy array of shape (n_samples, n_frames, 39)
    
    Returns:
        X_mirrored: mirrored version with same shape
    """
    # Define mapping for left-right pairs
    # Each body part has 3 coordinates (x, y, z)
    # Indices: 0-2: head, 3-5: left_shoulder, 6-8: left_elbow, 9-11: right_shoulder, etc.
    
    # Left and right body part index pairs (3 coordinates each)
    left_right_pairs = [
        (3, 9),   # left_shoulder (3-5) <-> right_shoulder (9-11)
        (6, 12),  # left_elbow (6-8) <-> right_elbow (12-14)
        (15, 18), # left_hand (15-17) <-> right_hand (18-20)
        (21, 24), # left_hip (21-23) <-> right_hip (24-26)
        (27, 30), # left_knee (27-29) <-> right_knee (30-32)
        (33, 36), # left_foot (33-35) <-> right_foot (36-38)
    ]
    
    X_mirrored = X.copy()
    
    for sample in range(X.shape[0]):
        for frame in range(X.shape[1]):
            for left_start, right_start in left_right_pairs:
                # Swap x, y, z coordinates (3 values per body part)
                left_indices = slice(left_start, left_start + 3)
                right_indices = slice(right_start, right_start + 3)
                
                # Swap left and right
                temp = X_mirrored[sample, frame, left_indices].copy()
                X_mirrored[sample, frame, left_indices] = X_mirrored[sample, frame, right_indices]
                X_mirrored[sample, frame, right_indices] = temp
    
    return X_mirrored

def augment_with_noise(X, noise_std=0.005):
    noise = np.random.normal(0, noise_std, X.shape)
    return X + noise