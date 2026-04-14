import os
import random
import pandas as pd
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import json
import copy
import mlflow
from sklearn.model_selection import KFold
from itertools import product
from sklearn.metrics import r2_score


# Function shuffles and splits files from a folder into 3 categories: training files, validation files and testing files
def split_csvfiles(datafolder, random_seed, training_prop, validation_prop):
    csv_files = []
    for f in os.listdir(datafolder):
        if f.endswith(".csv"):
            csv_files.append(f)

    random.seed(random_seed)
    random.shuffle(csv_files)

    train_n = int(len(csv_files) * training_prop)
    val_n = int(len(csv_files) * validation_prop)

    # Split
    if validation_prop == 0:
        train_files = csv_files[:train_n]
        test_files = csv_files[train_n:]

        return train_files, test_files

    else:
        train_files = csv_files[:train_n]
        val_files = csv_files[train_n: train_n + val_n]
        test_files = csv_files[train_n + val_n:]

        return train_files, val_files, test_files


# Function loads files into a dataframe
def load(files, data_dir):
    dataframes = []

    for f in files:
        path = os.path.join(data_dir, f)   
        df = pd.read_csv(path)

        # Get rid of trailing whitespace
        df.columns = df.columns.str.strip()           
        dataframes.append(df)             

    combined = pd.concat(dataframes, ignore_index=True)  # combine all

    return combined


# Function extracts the columns of interest from a dataframe and splits the
# data into input and target depending on if it's x,y data or x data
def input_target_split(dataframe):
    input_col_names = []
    target_col_names = []

    joints = ["head", "left_shoulder", "left_elbow", "right_shoulder", "right_elbow", "left_hand", "right_hand",
              "left_hip", "right_hip", "left_knee", "right_knee", "left_foot", "right_foot"]

    for joint in joints:
        input_col_names += [f"{joint}_x", f"{joint}_y"]
        target_col_names += [f"{joint}_z"]


    input_data = dataframe[input_col_names].copy()
    target_data = dataframe[target_col_names].copy()

    return input_data, target_data


# Function initializes the weights for a neural network model
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)


# Class blueprints a neural network model for determining the output x given
# we know the x, y coordinates
class ZPredictor(nn.Module):
    def __init__(self, hidden_layers: list, activation="relu", dropout=0.0):
        super().__init__()

        layers = []
        input_size = 26  # 13 joints x 2 (x, y)

        activations = {"relu": nn.ReLU(),
                       "tanh": nn.Tanh(),
                       "gelu": nn.GELU(),
                       "leaky_relu": nn.LeakyReLU()
                       }
        
        act = activations[activation]
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(act)

            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 13))  # 13 joints z output

        self.network = nn.Sequential(*layers)

        self.network.apply(init_weights)

    def forward(self, x):
        return self.network(x)


# Function builds a Z- predictor neural network model based on a configuration
def build_model(config, device):
    return ZPredictor(
        hidden_layers=config["hidden_layers"],
        activation=config["activation"],
        dropout=config["dropout"],
    ).to(device)


# Function computes metrics based on the true target data and the predicted
def compute_metrics(y_true, y_pred):
    mse = torch.mean((y_pred - y_true) ** 2).item()
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    bias = torch.mean(y_pred - y_true).item()

    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    r2 = r2_score(y_true_np, y_pred_np)

    return {"mse": mse, "mae": mae, "r2": r2, "bias": bias}


# Function evaluates a model's performance under a loss function
def evaluate_model(model, x_data, y_data, loss_fn):
    model.eval()
    with torch.no_grad():
        predictions = model(x_data)
        loss = loss_fn(predictions, y_data).item()
        metrics = compute_metrics(y_data, predictions)
    return loss, metrics, predictions


# Function trains a model and evaluates it on validation data
def train_one_model(model, config, x_train, y_train, x_val, y_val, verbose=False):
    optimizer_name = config["optimizer"]
    lr = config["learning_rate"]

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    loss_fn = nn.MSELoss()
    epochs = config["epochs"]
    patience = config["patience"]

    best_val_loss = float("inf")
    best_state = None
    history = []
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        train_predictions = model(x_train)
        train_loss = loss_fn(train_predictions, y_train)
        train_loss.backward()
        optimizer.step()

        # Evaluate
        val_loss, val_metrics, _ = evaluate_model(model, x_val, y_val, loss_fn)
        train_metrics = compute_metrics(y_train, train_predictions)

        row = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss.item()),
            "val_loss": float(val_loss),
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "val_r2": val_metrics["r2"],
        }
        history.append(row)


        # Early stoping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_loss, final_val_metrics, _ = evaluate_model(model, x_val, y_val, loss_fn)
    return {
        "model": model,
        "best_state": best_state,
        "history": history,
        "val_loss": final_val_loss,
        "val_metrics": final_val_metrics,
    }


# Function builds a model and has it training under k-fold cross validation
def cross_validation(config, X, Y, k, device):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_scores = []
    fold_models = []

    with mlflow.start_run():

        mlflow.log_params(config)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            x_train = X[train_idx]
            y_train = Y[train_idx]
            x_val = X[val_idx]
            y_val = Y[val_idx]

            model = build_model(config, device)
            model.apply(init_weights)

            results = train_one_model(model, config, x_train, y_train, x_val, y_val)

            fold_scores.append(results["val_loss"])
            fold_models.append(results["best_state"])

            mlflow.log_metric(f"fold_{fold}_val_loss ", results["val_loss"])

        avg_loss = sum(fold_scores) / len(fold_scores)

        mlflow.log_metric("cv_mean_val_loss", avg_loss)

    return {"cv_mean_loss": avg_loss,
            "fold_scores": fold_scores,
            "fold_models": fold_models}


# Function saves a candidate model into given folder
def save_candidate_model(model, model_name, candidates_dir):
    path = os.path.join(candidates_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), path)
    print(f"Saved candidate model: {path}")
    return path


# Function loads a model from a folder
def load_champion_info(metadata_dir):
    path = os.path.join(metadata_dir, "champion_info.json")

    if not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return None


# Function saves a champion model
def save_champion_model(model, model_name, mse, mae, hyperparameters, champion_dir, metadata_dir):
    model_path = os.path.join(champion_dir, "champion_model.pt")
    info_path = os.path.join(metadata_dir, "champion_info.json")

    # Save model weights
    torch.save(model.state_dict(), model_path)

    # Save metadata
    info = {
        "model_name": model_name,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mse": float(mse),
        "mae": float(mae),
        "hyperparameters": hyperparameters
    }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print("New champion model saved!")


# Functions updates the champion model if passed model has better mse
def update_champion(model, model_name, mse, mae, hyperparameters, champion_dir, metadata_dir):
    current = load_champion_info(metadata_dir)

    if current is None:
        print("No champion found --> saving first model")
        save_champion_model(model, model_name, mse, mae, hyperparameters, champion_dir, metadata_dir)

    elif mse < current["mse"]:
        print(f"New model is better (MSE {mse} < {current['mse']})")
        save_champion_model(model, model_name, mse, mae, hyperparameters, champion_dir, metadata_dir)

    else:
        print(f"Model NOT better (MSE {mse} ≥ {current['mse']})")