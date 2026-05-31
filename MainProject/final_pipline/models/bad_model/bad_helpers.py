import joblib
import numpy as np
import torch
import pandas as pd

from .bad_model import SquatClassifierCNN


USEFUL_JOINTS = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_foot", "right_foot"
]


def load_bad_model(model_path, device):
    """
    Loads the trained good/bad squat CNN model.
    """

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    model = SquatClassifierCNN(
        input_dim=(10, 24),
        dropout_rate=0.1
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_bad_scaler(scaler_path):
    """
    Loads the scaler used during good/bad model training.
    """

    return joblib.load(scaler_path)


def prepare_bad_input(fixed_c_df, scaler):
    """
    Prepares fixed-C data for the bad model.

    Input:
        fixed_c_df shape: (30, 39)

    Output:
        tensor shape: (1, 1, 10, 24)
    """

    df = fixed_c_df.copy()

    selected_cols = []

    for col in df.columns:
        if any(joint in col for joint in USEFUL_JOINTS):
            selected_cols.append(col)

    df = df[selected_cols]

    df = df.iloc[::3]

    X = df.values.astype(np.float32)

    if X.shape != (10, 24):
        raise ValueError(
            f"Bad model expected shape (10, 24), got {X.shape}"
        )

    X_flat = X.reshape(1, -1)

    X_flat_df = pd.DataFrame(X_flat, columns=scaler.feature_names_in_)

    X_scaled_flat = scaler.transform(X_flat_df)

    X_scaled = X_scaled_flat.reshape(1, 1, 10, 24)

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )

    return X_tensor


def predict_bad(fixed_c_df, model, scaler, device, threshold=0.5):
    """
    Predicts if the squat is bad or good.

    Returns:
        pred = 0 bad squat
        pred = 1 good squat
    """

    X = prepare_bad_input(
        fixed_c_df=fixed_c_df,
        scaler=scaler
    ).to(device)

    model.eval()

    with torch.no_grad():
        logits = model(X)
        prob = torch.sigmoid(logits).cpu().item()

    pred = int(prob >= threshold)

    return pred
