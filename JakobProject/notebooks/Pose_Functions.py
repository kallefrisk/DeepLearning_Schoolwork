import mediapipe as mp
import cv2
import json
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]


def data_to_json(data_path: str, save_path: str, model_path: str):
    data = extract_joint_nodes(data_path, model_path)
    save_to_json(data_path, save_path, data)


def extract_joint_nodes(data_path: str, model_path: str):
    static = not data_path.endswith(".mp4")

    base_options = python.BaseOptions(
        model_asset_path=model_path
    )

    results_list = []

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
                frame_joints = {}
                for idx, lm in enumerate(results.pose_landmarks[0]):
                    joint_name = LANDMARK_NAMES[idx]
                    frame_joints[joint_name] = {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility
                    }
                results_list.append({"frame": 0, "joints": frame_joints})

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

            rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
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
                    frame_joints = {}
                    for idx, lm in enumerate(results.pose_landmarks[0]):
                        joint_name = LANDMARK_NAMES[idx]
                        frame_joints[joint_name] = {
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z,
                            "visibility": lm.visibility
                        }
                    results_list.append({"frame": frame_idx, "joints": frame_joints})

                frame_idx += 1

            cap.release()

    return {"source": data_path, "frames": results_list}


def save_to_json(data_path: str, save_path: str, node_data: dict):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({
            "video_name": os.path.basename(data_path),
            "total_frames": len(node_data["frames"]),
            "frames": node_data["frames"]
        }, f, indent=2)
