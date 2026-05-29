import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import torch
from mediapipe_.mediapipe_helpers import extract_mediapipe_features
from models.ugly_models.ugly_helpers import load_ugly_model, load_ugly_scaler, predict_ugly
from models.trim_model.trim_helpers import load_trim_model, load_trim_scaler, drop_confidence_features, trim_sequence

if torch.backends.mps.is_available():
    device = torch.device("mps")      # Mac GPU (Apple Silicon)
elif torch.cuda.is_available():
    device = torch.device("cuda")     # Nvidia GPU
else:
    device = torch.device("cpu")


# =========================================
# Paths
# =========================================

BASE_DIR = Path(__file__).resolve().parent

video_path = BASE_DIR / "input_videos" / "score.avi"
# video_path = BASE_DIR / "input_videos" / "ugly.mov"
# video_path = BASE_DIR / "input_videos" / "bad.avi"

mediapipe_path = BASE_DIR / "mediapipe_" / "pose_landmarker.task"

ugly_model_path = BASE_DIR / "models" / "ugly_models" / "LSTMModel_checkpoint.pth"
ugly_scaler_path = BASE_DIR / "models" / "ugly_models" / "ugly_scaler.joblib"

trim_model_path = BASE_DIR / "models" / "trim_model" / "trim_model.pt"
trim_scaler_path = BASE_DIR / "models" / "trim_model" / "trim_scaler.joblib"

# =========================================
# Extract mediapipe
# =========================================
df = extract_mediapipe_features(video_path, mediapipe_path)

print(df.shape)
print(df.head())


# =========================================
# Ugly model
# =========================================
ugly_model = load_ugly_model(ugly_model_path, device)
ugly_scaler = load_ugly_scaler(ugly_scaler_path)

ugly_pred = predict_ugly(
    df=df,
    model=ugly_model,
    scaler=ugly_scaler,
    device=device,
    threshold=0.5
)

print("Ugly prediction:", ugly_pred)


# =========================================
# Drop irrelevenat features (visibility, precence, pose_score)
# =========================================
df_xyz = drop_confidence_features(df)


# =========================================
# Trim model
# =========================================

trim_model, trim_config = load_trim_model(trim_model_path, device)
trim_scaler = load_trim_scaler(trim_scaler_path)

trimmed_df, trim_preds = trim_sequence(
    df=df_xyz,
    model=trim_model,
    scaler=trim_scaler,
    device=device,
    seq_length=trim_config["seq_length"],
    stride=trim_config["stride"],
    threshold=0.5
)


print("Running mask: \n", trim_preds)
print("Trimmed shape:", trimmed_df.shape)

