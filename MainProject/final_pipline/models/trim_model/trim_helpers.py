import joblib
import numpy as np
import torch

from .trim_model import Recurrent_classifier


def load_trim_model(model_path, device):
    """
    Loads the trained trimming model checkpoint.
    """

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    config = checkpoint["hyperparameters"]

    model = Recurrent_classifier(
        hidden_layers=config["hidden_layers"],
        layer_type=config.get("layer_type", "LSTM"),
        dropout=config["dropout"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, config


def load_trim_scaler(scaler_path):
    """
    Loads the scaler used during trim model training.
    """

    return joblib.load(scaler_path)


def drop_confidence_features(df):
    """
    Removes visibility, presence, and pose_score features.
    Leaves only FrameNo and x/y/z features.
    """

    df = df.copy()

    drop_cols = [
        col for col in df.columns
        if col == "FrameNo"
        or col.endswith("_visibility")
        or col.endswith("_presence")
        or col == "pose_score"
    ]

    return df.drop(columns=drop_cols, errors="ignore")


def get_xyz_feature_columns(df):
    """
    Gets x, y, z feature columns in the dataframe.
    """

    return [
        col for col in df.columns
        if col.endswith("_x")
        or col.endswith("_y")
        or col.endswith("_z")
    ]


def predict_trim_mask(
    df,
    model,
    scaler,
    device,
    seq_length=30,
    stride=15,
    threshold=0.5
):
    """
    Predicts frame-level running/not-running mask.
    Returns probabilities and binary predictions.
    """

    df = df.copy()
    df.columns = df.columns.str.strip()

    if "FrameNo" not in df.columns:
        df["FrameNo"] = np.arange(len(df))

    feature_cols = get_xyz_feature_columns(df)

    X_np = df[feature_cols].values.astype(np.float32)

    X_scaled = scaler.transform(X_np)

    probs_sum = np.zeros(len(df))
    counts = np.zeros(len(df))

    model.eval()

    with torch.no_grad():
        for i in range(0, len(df) - seq_length + 1, stride):
            seq = X_scaled[i:i + seq_length]

            X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

            logits = model(X)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            probs_sum[i:i + seq_length] += probs
            counts[i:i + seq_length] += 1

    frame_probs = probs_sum / np.maximum(counts, 1)

    preds = (frame_probs >= threshold).astype(int)

    return frame_probs, preds


def trim_sequence(
    df,
    model,
    scaler,
    device,
    seq_length=30,
    stride=15,
    threshold=0.5
):
    """
    Trims dataframe between first and last predicted running frame.
    """

    df = df.copy()

    frame_probs, preds = predict_trim_mask(
        df=df,
        model=model,
        scaler=scaler,
        device=device,
        seq_length=seq_length,
        stride=stride,
        threshold=threshold
    )

    df["trim_probability"] = frame_probs
    df["pred_running"] = preds

    movement_frames = df[df["pred_running"] == 1]

    if len(movement_frames) == 0:
        return None, preds

    start_idx = movement_frames.index.min()
    stop_idx = movement_frames.index.max()

    trimmed_df = df.loc[start_idx:stop_idx].drop(
        columns=["trim_probability", "pred_running"],
        errors="ignore"
    )

    return trimmed_df, preds
