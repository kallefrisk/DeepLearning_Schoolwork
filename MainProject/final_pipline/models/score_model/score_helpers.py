import joblib
import numpy as np
import torch

from .score_model import ScoreCNN


def load_score_model(model_path, device):
    """
    Loads the trained score regression CNN model.
    """
    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    model = ScoreCNN(
        input_dim=(30, 39),
        dropout_rate=0.25
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_score_scaler(scaler_path):
    """
    Loads the scaler used during score model training.
    """

    return joblib.load(scaler_path)


def prepare_score_input(fixed_c_data, scaler):
    """
    Prepares fixed-C data for the score model.

    Input:
        fixed_c_data shape: (30, 39)

    Output:
        tensor shape: (1, 1, 30, 39)
    """

    if hasattr(fixed_c_data, "values"):
        X = fixed_c_data.values.astype(np.float32)
    else:
        X = np.asarray(fixed_c_data, dtype=np.float32)

    if X.shape != (30, 39):
        raise ValueError(
            f"Score model expected shape (30, 39), got {X.shape}"
        )

    # Scaler was trained frame-wise on 39 features
    X_flat = X.reshape(-1, 39)

    X_scaled_flat = scaler.transform(X_flat)

    X_scaled = X_scaled_flat.reshape(1, 1, 30, 39)

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )

    return X_tensor


def predict_score(fixed_c_data, model, scaler, device):
    """
    Predicts the squat score.

    Returns:
        score as float
    """

    X = prepare_score_input(
        fixed_c_data=fixed_c_data,
        scaler=scaler
    ).to(device)

    model.eval()

    with torch.no_grad():
        prediction = model(X).cpu().item()

    return prediction
