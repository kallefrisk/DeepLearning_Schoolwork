import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


JOINT_ORDER = [
    "head",
    "left_shoulder", "left_elbow",
    "right_shoulder", "right_elbow",
    "left_hand", "right_hand",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_foot", "right_foot"
]

LANDMARK_INDEX = {
    "head": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_hand": 15,
    "right_hand": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_foot": 27,
    "right_foot": 28,
}


def extract_mediapipe_world_features(video_path, model_path):
    """
    Extracts MediaPipe world landmarks from a video.
    Returns one row per frame.
    """

    base_options = python.BaseOptions(
        model_asset_path=str(model_path)
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    rows = []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30

        frame_idx = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_rgb
            )

            timestamp_ms = int((frame_idx / fps) * 1000)

            results = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            row = {"FrameNo": frame_idx}

            if results.pose_world_landmarks:
                landmarks = results.pose_world_landmarks[0]

                for joint in JOINT_ORDER:
                    idx = LANDMARK_INDEX[joint]
                    lm = landmarks[idx]

                    row[f"{joint}_x"] = lm.x
                    row[f"{joint}_y"] = lm.y
                    row[f"{joint}_z"] = lm.z
            else:
                for joint in JOINT_ORDER:
                    row[f"{joint}_x"] = 0.0
                    row[f"{joint}_y"] = 0.0
                    row[f"{joint}_z"] = 0.0

            rows.append(row)
            frame_idx += 1

        cap.release()

    return pd.DataFrame(rows)


def trim_world_df_with_preds(world_df, trim_preds):
    """
    Trims world landmark dataframe using trim predictions.
    """

    if len(world_df) != len(trim_preds):
        min_len = min(len(world_df), len(trim_preds))
        world_df = world_df.iloc[:min_len].copy()
        trim_preds = trim_preds[:min_len]

    world_df = world_df.copy()
    world_df["pred_running"] = trim_preds

    movement_frames = world_df[world_df["pred_running"] == 1]

    if len(movement_frames) == 0:
        return None

    start_idx = movement_frames.index.min()
    stop_idx = movement_frames.index.max()

    trimmed_world_df = world_df.loc[start_idx:stop_idx].drop(
        columns=["pred_running"],
        errors="ignore"
    )

    return trimmed_world_df


def center_around_hip(df):
    """
    Centers all joints around the midpoint between left and right hip.
    """

    df = df.copy()
    df.columns = df.columns.str.strip()

    for axis in ["_x", "_y", "_z"]:
        hip_mid = (
            df[f"left_hip{axis}"] +
            df[f"right_hip{axis}"]
        ) / 2

        axis_cols = [
            col for col in df.columns
            if col.endswith(axis)
        ]

        for col in axis_cols:
            df[col] = df[col] - hip_mid

    return df


def get_joint_positions(row):
    """
    Converts one dataframe row to 3D joint coordinates.
    """

    joints = {}

    for joint in JOINT_ORDER:
        x = row[f"{joint}_x"]
        y = row[f"{joint}_y"]
        z = row[f"{joint}_z"]

        # Rotate axes so the skeleton stands upright
        new_x = x
        new_y = z
        new_z = -y

        joints[joint] = np.array([new_x, new_y, new_z])

    return joints


def plot_skeleton(ax, joints):
    """
    Draws one skeleton frame.
    """

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

    ax.scatter(xs, ys, zs, c="red", s=50)

    for start, end in connections:
        ax.plot(
            [joints[start][0], joints[end][0]],
            [joints[start][1], joints[end][1]],
            [joints[start][2], joints[end][2]],
            c="red",
            linewidth=2
        )


def show_skeleton_animation(df):
    """
    Opens a Matplotlib window and shows the 3D skeleton animation.
    """

    df = center_around_hip(df)

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

    padding = 0.15

    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    z_range = max(z_max - z_min, 1e-6)

    def update(frame):
        ax.clear()

        row = df.iloc[frame]
        joints = get_joint_positions(row)

        plot_skeleton(ax, joints)

        ax.set_title(f"3D Skeleton - Frame {frame}")

        ax.set_xlim([
            x_min - padding * x_range,
            x_max + padding * x_range
        ])

        ax.set_ylim([
            y_min - padding * y_range,
            y_max + padding * y_range
        ])

        ax.set_zlim([
            z_min - padding * z_range,
            z_max + padding * z_range
        ])

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

    fig.anim = anim

    plt.show()

    return anim


def show_world_skeleton_for_video(video_path, model_path, trim_preds):
    """
    Full visualization pipeline:
    1. Extract world landmarks
    2. Trim with existing trim predictions
    3. Show 3D skeleton animation
    """

    world_df = extract_mediapipe_world_features(
        video_path,
        model_path
    )

    print("World dataframe shape:", world_df.shape)

    trimmed_world_df = trim_world_df_with_preds(
        world_df,
        trim_preds
    )

    if trimmed_world_df is None:
        print("Could not show skeleton: no movement frames found.")
        return None

    print("Trimmed world dataframe shape:", trimmed_world_df.shape)

    return show_skeleton_animation(trimmed_world_df)
