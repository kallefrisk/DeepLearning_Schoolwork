import os
import mediapipe as mp
import cv2
import json
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


Landmark_names = ["nose", "left_eye_inner", "left_eye", "left_eye_outer",
                  "right_eye_inner", "right_eye", "right_eye_outer",
                  "left_ear", "right_ear", "mouth_left", "mouth_right",
                  "left_shoulder", "right_shoulder", "left_elbow",
                  "right_elbow", "left_wrist", "right_wrist", "left_pinky",
                  "right_pinky", "left_index", "right_index", "left_thumb",
                  "right_thumb", "left_hip", "right_hip", "left_knee",
                  "right_knee", "left_ankle", "right_ankle", "left_heel",
                  "right_heel", "left_foot_index", "right_foot_index"]


def extract_joint_nodes(data_path: str, model_path: str):
    base_options = python.BaseOptions(model_asset_path=model_path)
    results_list = []

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(data_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_index = 0
        while cap.isOpened():
            read, frame = cap.read()
            if not read:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((frame_index/fps) * 1000)

            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_joints = {}
            if results.pose_landmarks:
                for index, landmark in enumerate(results.pose_landmarks[0]):
                    joint_name = Landmark_names[index]
                    frame_joints[joint_name] = {
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility
                    }

            results_list.append({"frame": frame_index, "joints": frame_joints})
            frame_index += 1
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
