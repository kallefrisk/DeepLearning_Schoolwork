import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import torch
from mediapipe_.mediapipe_helpers import extract_mediapipe_features
from models.ugly_models.ugly_helpers import load_ugly_model, load_ugly_scaler, predict_ugly
from models.trim_model.trim_helpers import load_trim_model, load_trim_scaler, drop_confidence_features, trim_sequence
from sequence_helpers import sequence_fixed_c
from models.bad_model.bad_helpers import load_bad_model, load_bad_scaler, predict_bad
from models.score_model.score_helpers import load_score_model, load_score_scaler, predict_score
from mediapipe_.world_visualization import show_world_skeleton_for_video

# =========================================
# Device
# =========================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# =========================================
# Paths
# =========================================

BASE_DIR = Path(__file__).resolve().parent

VIDEO_DIR = BASE_DIR / "input_videos"

mediapipe_path = BASE_DIR / "mediapipe_" / "pose_landmarker.task"

ugly_model_path = BASE_DIR / "models" / "ugly_models" / "ugly_model.pth"
ugly_scaler_path = BASE_DIR / "models" / "ugly_models" / "ugly_scaler.joblib"

trim_model_path = BASE_DIR / "models" / "trim_model" / "trim_model.pt"
trim_scaler_path = BASE_DIR / "models" / "trim_model" / "trim_scaler.joblib"

bad_model_path = BASE_DIR / "models" / "bad_model" / "bad_model.pt"
bad_scaler_path = BASE_DIR / "models" / "bad_model" / "bad_scaler.joblib"

score_model_path = BASE_DIR / "models" / "score_model" / "score_model.pt"
score_scaler_path = BASE_DIR / "models" / "score_model" / "score_scaler.joblib"


# =========================================
# Load all models once
# =========================================

print("Loading models...")

ugly_model = load_ugly_model(ugly_model_path, device)
ugly_scaler = load_ugly_scaler(ugly_scaler_path)

trim_model, trim_config = load_trim_model(trim_model_path, device)
trim_scaler = load_trim_scaler(trim_scaler_path)

bad_model = load_bad_model(bad_model_path, device)
bad_scaler = load_bad_scaler(bad_scaler_path)

score_model = load_score_model(score_model_path, device)
score_scaler = load_score_scaler(score_scaler_path)

print("All models loaded.\n")


# =========================================
# Helper functions
# =========================================

def get_video_files(video_dir):
    """
    Returns all supported video files in the selected folder.
    """

    video_extensions = [".mp4", ".mov", ".avi", ".mkv"]

    return sorted([
        file for file in video_dir.iterdir()
        if file.suffix.lower() in video_extensions
    ])


def choose_video(video_dir):
    """
    Shows a simple terminal menu for choosing a video.
    """

    videos = get_video_files(video_dir)

    if len(videos) == 0:
        print("No videos found in input_videos folder.")
        return None

    print("\n==============================")
    print("Choose a video")
    print("==============================")

    for i, video in enumerate(videos, start=1):
        print(f"{i}. {video.name}")

    print("0. Exit")

    choice = input("\nEnter video number: ")

    if choice == "0":
        return "exit"

    if not choice.isdigit():
        print("Invalid input. Please enter a number.")
        return None

    choice = int(choice)

    if choice < 1 or choice > len(videos):
        print("Invalid video number.")
        return None

    return videos[choice - 1]


def run_pipeline(video_path):
    """
    Runs the full squat analysis pipeline on one selected video.
    """

    print("\n==============================")
    print(f"Processing: {video_path.name}")
    print("==============================")

    # -----------------------------------------
    # MediaPipe
    # -----------------------------------------

    df = extract_mediapipe_features(
        video_path,
        mediapipe_path
    )

    print(f"MediaPipe dataframe shape: {df.shape}")

    # -----------------------------------------
    # Ugly model
    # -----------------------------------------

    ugly_pred = predict_ugly(
        df=df,
        model=ugly_model,
        scaler=ugly_scaler,
        device=device,
        threshold=0.5
    )

    print("\nUgly prediction:", ugly_pred)

    if ugly_pred == 0:
        print("\nERROR: Recording rejected. The video quality is too poor.")
        return

    # -----------------------------------------
    # Trim model
    # -----------------------------------------

    df_xyz = drop_confidence_features(df)

    trimmed_df, trim_preds = trim_sequence(
        df=df_xyz,
        model=trim_model,
        scaler=trim_scaler,
        device=device,
        seq_length=trim_config["seq_length"],
        stride=trim_config["stride"],
        threshold=0.5
    )

    if trimmed_df is None:
        print("\nERROR: No squat movement detected.")
        return

    # -----------------------------------------
    # Fixed C
    # -----------------------------------------

    try:
        fixed_c_df = sequence_fixed_c(trimmed_df)
    except ValueError as error:
        print(f"\nERROR: {error}")
        return

    # -----------------------------------------
    # Bad model
    # -----------------------------------------

    bad_pred = predict_bad(
        fixed_c_df=fixed_c_df,
        model=bad_model,
        scaler=bad_scaler,
        device=device,
        threshold=0.5
    )

    print("Bad model prediction:", bad_pred)

    if bad_pred == 0:
        print("\nERROR: Squat rejected. The squat is classified as bad.")
        return

    # -----------------------------------------
    # Score model
    # -----------------------------------------

    score = predict_score(
        fixed_c_data=fixed_c_df,
        model=score_model,
        scaler=score_scaler,
        device=device
    )

    print("\n==============================")
    print("Final result")
    print("==============================")
    print("Recording quality: accepted")
    print("Squat quality: accepted")
    print(f"Squat score: {score:.2f}")

    show_skeleton = input("\nShow 3D skeleton? (y/n): ")

    if show_skeleton.lower() == "y":
        show_world_skeleton_for_video(
            video_path=video_path,
            model_path=mediapipe_path,
            trim_preds=trim_preds)


# =========================================
# Main loop
# =========================================

while True:
    selected_video = choose_video(VIDEO_DIR)

    if selected_video == "exit":
        print("\nExiting program.")
        break

    if selected_video is None:
        continue

    try:
        run_pipeline(selected_video)
    except Exception as error:
        print(f"\nUnexpected error: {error}")

    input("\nPress Enter to choose another video...")
