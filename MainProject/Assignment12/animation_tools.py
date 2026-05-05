import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def get_joint_positions(row):
    joints = {}

    joint_names = [
        "head", "left_shoulder", "left_elbow", "right_shoulder", "right_elbow",
        "left_hand", "right_hand", "left_hip", "right_hip", "left_knee",
        "right_knee", "left_foot", "right_foot"
    ]

    for joint in joint_names:
        x = row[f"{joint}_x"]
        y = row[f"{joint}_y"]
        z = row[f"{joint}_z"]

        joints[joint] = np.array([x, y, z])

    return joints


def plot_skeleton(ax, joints, color="red", alpha=1.0):
    connections = [
        ("head", "left_shoulder"),
        ("head", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_hand"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_hand"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_foot"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_foot"),
    ]

    xs = [joints[joint][0] for joint in joints]
    ys = [joints[joint][1] for joint in joints]
    zs = [joints[joint][2] for joint in joints]

    ax.scatter(xs, ys, zs, c=color, s=50, alpha=alpha)

    for start, end in connections:
        x_line = [joints[start][0], joints[end][0]]
        y_line = [joints[start][1], joints[end][1]]
        z_line = [joints[start][2], joints[end][2]]
        ax.plot(x_line, y_line, z_line, c=color, linewidth=2, alpha=alpha)


def center_around_hip(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    for axis in ("_x", "_y", "_z"):
        left_hip = df[f"left_hip{axis}"]
        right_hip = df[f"right_hip{axis}"]
        hip_mid = (left_hip + right_hip) / 2
        axis_cols = [c for c in df.columns if c.endswith(axis)]
        for col in axis_cols:
            df[col] -= hip_mid

    return df


def create_skeleton_animation(df):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    all_joints = []
    for _, row in df.iterrows():
        joints = get_joint_positions(row)
        all_joints.extend(joints.values())

    all_joints = np.array(all_joints)

    x_min, x_max = all_joints[:, 0].min(), all_joints[:, 0].max()
    y_min, y_max = all_joints[:, 1].min(), all_joints[:, 1].max()
    z_min, z_max = all_joints[:, 2].min(), all_joints[:, 2].max()

    padding = 0.1

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    def update(frame):
        ax.clear()

        row = df.iloc[frame]
        joints = get_joint_positions(row)

        plot_skeleton(ax, joints, color="red", alpha=1.0)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"Skeleton Pose - Frame {frame}")

        ax.set_xlim([x_min - padding * x_range, x_max + padding * x_range])
        ax.set_ylim([y_min - padding * y_range, y_max + padding * y_range])
        ax.set_zlim([z_min - padding * z_range, z_max + padding * z_range])

        ax.set_box_aspect([x_range, y_range, z_range])
        ax.grid(True, alpha=0.3)

        return ax,

    anim = FuncAnimation(
        fig,
        update,
        frames=len(df),
        interval=50,
        blit=False,
        repeat=True
    )

    plt.close(fig)
    return anim


def animate_df(df):
    df_centered = center_around_hip(df)

    print(f"Loaded {len(df_centered)} frames of skeleton data")
    print("Creating animation...")

    anim = create_skeleton_animation(df_centered)

    print("Animation created successfully!")
    return anim