import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def extract_mediapipe_features(video_path, model_path):
    """
    Runs MediaPipe Pose Landmarker on a video and returns a dataframe.

    Output:
        One row per video frame.

    Features:
        FrameNo
        joint_x
        joint_y
        joint_z
        joint_visibility
        joint_presence
        pose_score
    """

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

    def create_empty_row(frame_idx):
        """
        Creates a zero-filled row when MediaPipe does not detect a pose.
        """

        row = {"FrameNo": frame_idx}

        for joint in JOINT_ORDER:
            row[f"{joint}_x"] = 0.0
            row[f"{joint}_y"] = 0.0
            row[f"{joint}_z"] = 0.0
            row[f"{joint}_visibility"] = 0.0
            row[f"{joint}_presence"] = 0.0

        row["pose_score"] = 0.0

        return row

    def create_pose_row(frame_idx, landmarks):
        """
        Creates one dataframe row from detected MediaPipe landmarks.
        """

        row = {"FrameNo": frame_idx}

        visibility_values = []
        presence_values = []

        for joint in JOINT_ORDER:
            idx = LANDMARK_INDEX[joint]
            lm = landmarks[idx]

            visibility = getattr(lm, "visibility", 0.0)
            presence = getattr(lm, "presence", 0.0)

            row[f"{joint}_x"] = lm.x
            row[f"{joint}_y"] = lm.y
            row[f"{joint}_z"] = lm.z
            row[f"{joint}_visibility"] = visibility
            row[f"{joint}_presence"] = presence

            visibility_values.append(visibility)
            presence_values.append(presence)

        row["pose_score"] = np.mean(
            visibility_values + presence_values
        )

        return row

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

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            timestamp_ms = int((frame_idx / fps) * 1000)

            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.pose_landmarks:
                row = create_pose_row(
                    frame_idx,
                    results.pose_landmarks[0]
                )
            else:
                row = create_empty_row(frame_idx)

            rows.append(row)

            frame_idx += 1

        cap.release()

    df = pd.DataFrame(rows)

    return df