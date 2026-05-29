from pathlib import Path
import sys
import joblib
import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:

    sys.path.insert(0, str(CURRENT_DIR))

from .RNN_models import SimpleRNNModel, LSTMModel, GRUModel


def load_ugly_model(model_path, device):
    """
    Loads the trained ugly-recording model checkpoint.
    """

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    config = checkpoint["model_config"]

    model_type = config["model_type"]

    model = model_type(
        config["input_size"],
        config["hidden_size"],
        config["depth"],
        config["num_of_classification_labels"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def load_ugly_scaler(scaler_path):
    """
    Loads the scaler used when training the ugly model.
    """

    return joblib.load(scaler_path)

def prepare_ugly_input(df, scaler, num_start_frames=30, step=3):
    """
    Prepares MediaPipe dataframe for ugly model.

    Output shape:
        (1, 10, 66)
    """

    df = df.copy()

    if "FrameNo" in df.columns:
        df = df.drop(columns=["FrameNo"])

    if len(df) < num_start_frames:
        raise ValueError(
            f"Video too short for ugly model: {len(df)} frames"
        )

    df_start = df.iloc[:num_start_frames]

    df_selected = df_start.iloc[::step]

    X = df_selected.values.astype(np.float32)

    X_flat = X.reshape(1, -1)

    X_scaled_flat = scaler.transform(X_flat)

    X_scaled = X_scaled_flat.reshape(1, 10, 66)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    return X_tensor


def predict_ugly(df, model, scaler, device, threshold=0.5):
    """
    Predicts if the recording is ugly.

    Returns:
        pred = 0 means reject
        pred = 1 means continue
        prob = model probability
    """

    X = prepare_ugly_input(
        df=df,
        scaler=scaler
    ).to(device)

    model.eval()

    with torch.no_grad():
        logits = model(X)
        prob = torch.sigmoid(logits).cpu().item()

    pred = int(prob >= threshold)

    return pred
