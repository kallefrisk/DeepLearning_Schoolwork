import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import copy

def train_one_model(model, config, x_train, y_train, x_val, y_val, loss_fn):

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    epochs = config["epochs"]

    best_val_mse = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    patience = 10
    epochs_no_improve = 0

    # Convergence tracking
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mse": []
    }

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        predictions = model(x_train)

        train_loss = loss_fn(predictions, y_train)
        train_loss.backward()
        optimizer.step()

        # Validation with mse as metric
        model.eval()
        with torch.no_grad():
            val_predictions = model(x_val)
            val_loss = loss_fn(val_predictions, y_val)

            val_mse = val_loss.item()


        # Store history for each epoch
        history["train_loss"].append(train_loss.item())
        history["val_loss"].append(val_loss.item())
        history["val_mse"].append(val_mse)

        # Early stopping
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    # Load best weights
    model.load_state_dict(best_state)

    return best_val_mse, model, history


# Cross validation
def cross_validate_model(config, X, y, loss_fn, input_size, n_splits=10):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_mse_scores = []
    fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # --Augmentate for each individual fold to avoid leakage--
        # Mirror augmentation
        X_train = pd.concat([X_train, mirror_dataframe(X_train)])
        y_train = np.concatenate([y_train, y_train])

        # Noise augmentation
        X_train = pd.concat([X_train, augment_with_noise(X_train)])
        y_train = np.concatenate([y_train, y_train])

        # --Scale for each fold  to avoid data leakage-- (df -> np.array)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        # Convert the y array to numpy arrays too
        y_train = np.asarray(y_train)
        y_val   = np.asarray(y_val)

        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_val   = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device).view(-1, 1)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(device).view(-1, 1)

        # Build fresh model
        model = build_dense_model(config, input_size)

        # Train
        val_mse, model, history = train_one_model(
            model, config,
            X_train, y_train,
            X_val, y_val,
            loss_fn
        )

        fold_mse_scores.append(val_mse)
        fold_histories.append(history)


    # Aggregate results
    mean_mse = np.mean(fold_mse_scores)
    std_mse  = np.std(fold_mse_scores)

    return {
        "cv_mean_mse": mean_mse,
        "mse_std": std_mse,
        "fold_scores": fold_mse_scores
    }