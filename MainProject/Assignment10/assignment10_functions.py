import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt


def extract_mediapipe_to_csv(data_path: str, save_path: str, model_path: str):

    video_extensions = [".mp4", ".mov", ".avi", ".mkv"]
    static = not any(data_path.lower().endswith(ext) for ext in video_extensions)

    base_options = python.BaseOptions(model_asset_path=model_path)

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

    data = []

    if static:
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5
        )

        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            image = cv2.imread(data_path)
            if image is None:
                raise ValueError(f"Could not read image: {data_path}")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = landmarker.detect(mp_image)

            if results.pose_landmarks:
                row = {"FrameNo": 0}

                for joint in JOINT_ORDER:
                    idx = LANDMARK_INDEX[joint]
                    lm = results.pose_landmarks[0][idx]

                    row[f"{joint}_x"] = lm.x
                    row[f"{joint}_y"] = lm.y
                    row[f"{joint}_z"] = lm.z

                data.append(row)

    else:
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(data_path)

            if not cap.isOpened():
                raise ValueError(f"Could not open video: {data_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

                timestamp_ms = int((frame_idx / fps) * 1000)
                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                if results.pose_landmarks:
                    row = {"FrameNo": frame_idx}

                    for joint in JOINT_ORDER:
                        idx = LANDMARK_INDEX[joint]
                        lm = results.pose_landmarks[0][idx]

                        row[f"{joint}_x"] = lm.x
                        row[f"{joint}_y"] = lm.y
                        row[f"{joint}_z"] = lm.z

                    data.append(row)

                frame_idx += 1

            cap.release()

    # ---- SAVE CSV ----
    columns = ["FrameNo"]
    for joint in JOINT_ORDER:
        columns += [f"{joint}_x", f"{joint}_y", f"{joint}_z"]

    df = pd.DataFrame(data)
    df = df[columns]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"Saved {len(df)} rows to {save_path}")




def extract_mediapipe_to_csv_world(data_path: str, save_path: str, model_path: str):

    video_extensions = [".mp4", ".mov", ".avi", ".mkv"]
    static = not any(data_path.lower().endswith(ext) for ext in video_extensions)

    base_options = python.BaseOptions(model_asset_path=model_path)

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

    data = []

    if static:
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5
        )

        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            image = cv2.imread(data_path)
            if image is None:
                raise ValueError(f"Could not read image: {data_path}")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = landmarker.detect(mp_image)

            if results.pose_world_landmarks:
                row = {"FrameNo": 0}

                for joint in JOINT_ORDER:
                    idx = LANDMARK_INDEX[joint]
                    lm = results.pose_world_landmarks[0][idx]

                    row[f"{joint}_x"] = lm.x
                    row[f"{joint}_y"] = lm.y
                    row[f"{joint}_z"] = lm.z

                data.append(row)

    else:
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(data_path)

            if not cap.isOpened():
                raise ValueError(f"Could not open video: {data_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

                timestamp_ms = int((frame_idx / fps) * 1000)
                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                if results.pose_world_landmarks:
                    row = {"FrameNo": frame_idx}

                    for joint in JOINT_ORDER:
                        idx = LANDMARK_INDEX[joint]
                        lm = results.pose_world_landmarks[0][idx]

                        row[f"{joint}_x"] = lm.x
                        row[f"{joint}_y"] = lm.y
                        row[f"{joint}_z"] = lm.z

                    data.append(row)

                frame_idx += 1

            cap.release()

    # ---- SAVE CSV ----
    columns = ["FrameNo"]
    for joint in JOINT_ORDER:
        columns += [f"{joint}_x", f"{joint}_y", f"{joint}_z"]

    df = pd.DataFrame(data)
    df = df[columns]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"Saved {len(df)} rows to {save_path}")


def euclidean_2d(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def euclidean_3d(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def add_3D_distances(df):
    df = df.copy()
    df["hip_to_shoulder"] = euclidean_3d(df["left_hip_x"], df["left_hip_y"], df["left_hip_z"],
                                         df["left_shoulder_x"], df["left_shoulder_y"], df["left_shoulder_z"])
    df["knee_to_hip"] = euclidean_3d(df["left_knee_x"], df["left_knee_y"], df["left_knee_z"],
                                     df["left_hip_x"], df["left_hip_y"], df["left_hip_z"])
    df["hip_to_hip"] = euclidean_3d(df["left_hip_x"], df["left_hip_y"], df["left_hip_z"],
                                    df["right_hip_x"], df["right_hip_y"], df["right_hip_z"])
    df["knee_to_ankle"] = euclidean_3d(df["left_knee_x"], df["left_knee_y"], df["left_knee_z"],
                                       df["left_foot_x"], df["left_foot_y"], df["left_foot_z"])
    df["shoulder_to_shoulder"] = euclidean_3d(df["left_shoulder_x"], df["left_shoulder_y"], df["left_shoulder_z"],
                                              df["right_shoulder_x"], df["right_shoulder_y"], df["right_shoulder_z"])
    return df


def add_2D_distances(df):
    df = df.copy()
    df["hip_to_shoulder"]        = euclidean_2d(df["left_hip_x"], df["left_hip_y"],
                                        df["left_shoulder_x"], df["left_shoulder_y"])
    df["knee_to_hip"]            = euclidean_2d(df["left_knee_x"], df["left_knee_y"],
                                        df["left_hip_x"], df["left_hip_y"])
    df["hip_to_hip"]             = euclidean_2d(df["left_hip_x"], df["left_hip_y"],
                                        df["right_hip_x"], df["right_hip_y"])
    df["knee_to_ankle"]          = euclidean_2d(df["left_knee_x"], df["left_knee_y"],
                                        df["left_foot_x"], df["left_foot_y"])
    df["shoulder_to_shoulder"]   = euclidean_2d(df["left_shoulder_x"], df["left_shoulder_y"],
                                        df["right_shoulder_x"], df["right_shoulder_y"])
    return df


def convert_to_pixel_coordinates(df, frame_width, frame_height):

    joints = [
        "head",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_hand", "right_hand",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_foot", "right_foot"]

    df = df.copy()

    for joint in joints:
        df[f"{joint}_x"] = df[f"{joint}_x"] * frame_width
        df[f"{joint}_y"] = df[f"{joint}_y"] * frame_height
    return df


def align_by_frame(mp_df, kinect_df):
    kinect_frames = kinect_df["FrameNo"].values
    mp_aligned = mp_df[mp_df["FrameNo"].isin(kinect_frames)]
    mp_aligned = mp_aligned.reset_index(drop=True)
    kinect_aligned = kinect_df.reset_index(drop=True)

    return mp_aligned, kinect_aligned


def print_comparison_table(avg, ground_truth, scale, reference):

    all_errors = []

    print(f"{'Limb':<25} {'Tape (cm)':>10} {'MediaPipe (cm)':>15} {'Error (cm)':>12} {'Error (%)':>10}")
    print("-" * 75)

    for key, true_val in ground_truth.items():
        mp_val = avg[key] * scale
        error = abs(mp_val - true_val)
        error_pct = (error / true_val) * 100 if true_val != 0 else 0
        if key != reference:
            all_errors.append(error)
        print(f"{key:<25} {true_val:>10.1f} {mp_val:>15.1f} {error:>12.1f} {error_pct:>10.1f}%")

    print("-" * 75)
    print(f"{'Average error':<25} {'':>10} {'':>15} {np.mean(all_errors):>12.1f}")


def print_kinect_mp_comparison(kinect_dist, mp_scaled, errors):

    distance_names = [
        "hip_to_shoulder",
        "knee_to_hip",
        "hip_to_hip",
        "knee_to_ankle",
        "shoulder_to_shoulder"]

    print(f"{'Measurement':<25} {'Kinect':>12} {'MediaPipe':>12} {'Error':>12} {'Error %':>10}")
    print("-" * 75)

    all_errors = []

    for col in distance_names:
        kinect_avg = kinect_dist[col].mean()
        mp_avg = mp_scaled[col].mean()
        error_avg = errors[col].mean()
        error_pct = (error_avg / kinect_avg) * 100 if kinect_avg != 0 else 0
        all_errors.append(error_avg)

        print(f"{col:<25} {kinect_avg * 100:>12.1f} {mp_avg * 100:>12.1f} {error_avg * 100:>12.1f} {error_pct:>10.1f}%")

    print("-" * 75)
    print(f"{'Average error':<25} {'':>12} {'':>12} {np.mean(all_errors) * 100:>12.1f}")
