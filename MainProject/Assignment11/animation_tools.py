import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def read_skeleton_data(file_path):
    """Read skeleton data from CSV file"""
    df = pd.read_csv(file_path)
    return df

def get_joint_positions(row):
    """Extract joint positions from a dataframe row"""
    joints = {}
    
    # Define joint names and their column prefixes
    joint_names = [
        'head', 'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
        'left_hand', 'right_hand', 'left_hip', 'right_hip', 'left_knee', 
        'right_knee', 'left_foot', 'right_foot'
    ]
    
    for joint in joint_names:
        x = row[f'{joint}_x']
        y = row[f'{joint}_y']
        z = row[f'{joint}_z']
        joints[joint] = np.array([x, y, z])
    
    return joints

def plot_skeleton(ax, joints, color='blue', alpha=1.0):
    """Plot a single skeleton frame"""
    
    # Define connections between joints
    connections = [
        ('head', 'left_shoulder'),
        ('head', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_hand'),
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_hand'),
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_foot'),
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_foot'),
    ]
    
    # Plot joints as scatter points
    xs = [joints[joint][0] for joint in joints]
    ys = [joints[joint][1] for joint in joints]
    zs = [joints[joint][2] for joint in joints]
    ax.scatter(xs, ys, zs, c=color, s=50, alpha=alpha)
    
    # Plot connections as lines
    for start, end in connections:
        if start in joints and end in joints:
            x_line = [joints[start][0], joints[end][0]]
            y_line = [joints[start][1], joints[end][1]]
            z_line = [joints[start][2], joints[end][2]]
            ax.plot(x_line, y_line, z_line, c=color, linewidth=2, alpha=alpha)

def create_skeleton_animation(df, save_folder, output_file='skeleton_animation.gif'):
    """Create animated 3D skeleton visualization"""
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get min/max values for consistent axis limits
    all_joints = []
    for idx, row in df.iterrows():
        joints = get_joint_positions(row)
        all_joints.extend(joints.values())
    
    all_joints = np.array(all_joints)
    x_min, x_max = all_joints[:, 0].min(), all_joints[:, 0].max()
    y_min, y_max = all_joints[:, 1].min(), all_joints[:, 1].max()
    z_min, z_max = all_joints[:, 2].min(), all_joints[:, 2].max()
    
    # Add padding
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    def update(frame):
        ax.clear()
        
        # Get current frame's joints
        row = df.iloc[frame]
        joints = get_joint_positions(row)
        
        # Plot skeleton
        plot_skeleton(ax, joints, color='red', alpha=1.0)
        
        # Set labels and title
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title(f'Skeleton Pose - Frame {frame}', fontsize=12)
        
        # Set consistent axis limits
        ax.set_xlim([x_min - padding * x_range, x_max + padding * x_range])
        ax.set_ylim([y_min - padding * y_range, y_max + padding * y_range])
        ax.set_zlim([z_min - padding * z_range, z_max + padding * z_range])
        
        # Set equal aspect ratio
        ax.set_box_aspect([x_range, y_range, z_range])
        
        # Add a simple grid for better depth perception
        ax.grid(True, alpha=0.3)
        
        return ax,
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(df), interval=50, blit=False, repeat=True)
    
    # Save animation
    anim.save(os.path.join(save_folder, output_file), writer='pillow', fps=20)
    print(f"Animation saved as {output_file}")
    
    return anim

def plot_multiple_frames(df, save_folder, frame_indices=None, output_file='skeleton_multiframe.png'):
    """Plot multiple skeleton frames in one figure"""
    
    if frame_indices is None:
        # Default: show first 4 frames
        frame_indices = [0, 1, 2, 3] if len(df) >= 4 else list(range(len(df)))
    
    n_frames = len(frame_indices)
    fig = plt.figure(figsize=(15, 12))
    
    # Calculate subplot grid
    cols = min(2, n_frames)
    rows = (n_frames + cols - 1) // cols
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        
        row = df.iloc[frame_idx]
        joints = get_joint_positions(row)
        
        plot_skeleton(ax, joints, color=colors[i % len(colors)], alpha=1.0)
        
        ax.set_xlabel('X (m)', fontsize=8)
        ax.set_ylabel('Y (m)', fontsize=8)
        ax.set_zlabel('Z (m)', fontsize=8)
        ax.set_title(f'Frame {frame_idx}', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, output_file), dpi=150, bbox_inches='tight')
    print(f"Multi-frame plot saved as {output_file}")
    plt.show()

def animate(path: str, save_folder_path: str = "./plots"):
    """Main function to run the skeleton visualization"""
    
    # Create sample data (replace this with your actual data loading)
    # For demonstration, I'll create a small sample based on your data
    
    # If you have a CSV file, use this:
    # df = read_skeleton_data('your_skeleton_data.csv')
    
    # For demonstration with your provided data:
   
    df = pd.read_csv(path)
    df = df.rename(columns={" head_x": "head_x"})

    for axis in ("_x", "_y", "_z"):
        left_hip = df[f"left_hip{axis}"]
        right_hip = df[f"right_hip{axis}"]
        hip_mid = (left_hip + right_hip) / 2
        axis_cols = [c for c in df.columns if c.endswith(axis)]
        for col in axis_cols:
            df[col] -= hip_mid
    
    print(f"Loaded {len(df)} frames of skeleton data")
    
    # Option 1: Show static plot of multiple frames
    # print("\nGenerating multi-frame plot...")
    # plot_multiple_frames(df, save_folder=save_folder_path, frame_indices=[0, 1, 2, 3], output_file='skeleton_frames.png')
    
    # Option 2: Create animated visualization
    print("\nCreating animation...")
    try:
        anim = create_skeleton_animation(df, save_folder=save_folder_path, output_file='skeleton_animation.gif')
        print("Animation created successfully!")
    except Exception as e:
        print(f"Could not create animation: {e}")
        print("Try installing: pip install pillow")