from pathlib import Path
from mediapipe_.mediapipe_helpers import extract_mediapipe_features


# =========================================
# Paths
# =========================================

BASE_DIR = Path(__file__).resolve().parent

video_path = BASE_DIR / "input_videos" / "score.avi"
model_path = BASE_DIR / "mediapipe_" / "pose_landmarker.task"


# =========================================
# Extract mediapipe
# =========================================
df = extract_mediapipe_features(video_path, model_path)

print(df.shape)
print(df.head())
