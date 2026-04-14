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
import mlflow
from sklearn.model_selection import KFold
from itertools import product


def split_csvfiles(datafolder, random_seed, training_prop, validation_prop):
    csv_files = []
    for f in os.listdir(datafolder):
        if f.endswith(".csv"):
            csv_files.append(f)
    
    random.seed(random_seed)
    random.shuffle(csv_files)

    # Train/Validation/Test with proportions 80/10/10
    train_n = int(len(csv_files) * training_prop)
    val_n = int(len(csv_files) * validation_prop)
    test_n = len(csv_files) - train_n - val_n

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


# Enis function but it doesn't rely on a global candidates_dir variable
def save_candidate_model(model, model_name, candidates_dir):
    path = os.path.join(candidates_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), path)
    print(f"Saved candidate model: {path}")
    return path


# Enis load_champion function but doesn't rely on global metadata_dir variable
def load_champion_info(metadata_dir):
    path = os.path.join(metadata_dir, "champion_info.json")

    if not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return None


# Enis save_champion model but function doesn't rely on globally defined dirs
def save_champion_model(champion_dir, metadata_dir, model, model_name, mse, mae, hyperparameters):
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


# Enis update_champion but with local variable dirs
def update_champion(metadata_dir, champion_dir, model, model_name, mse, mae, hyperparameters):
    current = load_champion_info(metadata_dir)

    if current is None:
        print("No champion found --> saving first model")
        save_champion_model(champion_dir, metadata_dir, model, model_name, mse, mae, hyperparameters)

    elif mae < current["mae"]:
        print(f"New model is better (MSE {mse} < {current['mse']})")
        save_champion_model(champion_dir, metadata_dir, model, model_name, mse, mae, hyperparameters)

    else:
        print(f"Model NOT better (MSE {mse} ≥ {current['mse']})")


def init_weights(m):
    if isinstance(m, nn.Linear):
        # nn.init.xavier_uniform_(m.weight) # good for Tanh
        nn.init.kaiming_uniform_(m.weight)  # good for ReLU
        nn.init.zeros_(m.bias)


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


def build_model(config):
    return ZPredictor(hidden_layers=config["layers"], activation=config["activation"], dropout=config["dropout"]).to(device)


# Modified version of training logic by Karl or Jakob but suitable for cross validation
def train_one_model(model, config, x_train, y_train, x_val, y_val, loss_fn):

    optimizer = optim.Adam(model.parameters(), lr = config["lr"])
    epochs = config["epochs"]

    best_val_loss = float("inf")
    best_state = None
    patience = 10
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Train step
        model.train()
        optimizer.zero_grad()
        predictions = model(x_train)
        train_loss = loss_fn(predictions, y_train)
        train_loss.backward()
        optimizer.step()

        # Evaluation step
        model.eval()
        with torch.no_grad():
            val_predictions = model(x_val)
            val_loss = loss_fn(val_predictions, y_val)

            # val_mae = torch.mean(torch.abs(val_predictions - y_val))
            # val_mse = torch.mean(torch.square(val_predictions - y_val))
            # val_r2 = r2_score(y_val.cpu(), val_predictions.cpu())
            # val_bias = torch.mean(val_predictions - y_val)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    return best_val_loss, best_state


def cross_validation(config, X, Y, k):
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

            model = build_model(config)
            model.apply(init_weights)

            val_loss, best_state = train_one_model(model, config, x_train, y_train, x_val, y_val, loss_fn=nn.MSELoss())

            fold_scores.append(val_loss)
            fold_models.append(best_state)

            mlflow.log_metric(f"fold_{fold}_val_loss ", val_loss)
        
        avg_loss = sum(fold_scores) / len(fold_scores)

        mlflow.log_metric("cv_mean_val_loss", avg_loss)
    
    return {"cv_mean_loss": avg_loss,
            "fold_scores": fold_scores,
            "fold_models": fold_models}





if torch.backends.mps.is_available():
    device = torch.device("mps")      # Mac GPU (Apple Silicon)
    print("Apple GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")     # Nvidia GPU / AMD ROCm GPU
    print("Nvidia/AMD GPU")
else:
    device = torch.device("cpu")
    print("CPU")


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("z_prediction")


datafolder = "MainProject/Assignment9/data/kinect_good_preprocessed"
random_seed = 42

# Alt 1: 80/10/10 split
# train_files, val_files, test_files = split_csvfiles(datafolder, random_seed, 0.8, 0.1)
# val_data = load(val_files, datafolder)

# Alt 2: 90/10 since we don't need validation data if we run k-fold cv
train_files, test_files = split_csvfiles(datafolder, random_seed, 0.9, 0)

train_data = load(train_files, datafolder)
test_data = load(test_files, datafolder)

print(train_data.columns.tolist())


x_train, y_train = input_target_split(train_data)
# x_val, y_val = input_target_split(val_data)
x_test, y_test = input_target_split(test_data)


# Define paths
model_dir = "../assignment9_models"
candidates_dir = os.path.join(model_dir, "candidates")
champion_dir = os.path.join(model_dir, "champion")
metadata_dir = os.path.join(model_dir, "metadata")


# Convert data to tensors
X_train = torch.tensor(x_train.values, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.values, dtype=torch.float32).to(device)
# X_val = torch.tensor(x_val.values, dtype=torch.float32).to(device)
# y_val = torch.tensor(y_val.values, dtype=torch.float32).to(device)
X_test = torch.tensor(x_test.values, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test.values, dtype=torch.float32).to(device)


# Define search space 
param_grid = {"layers": [[256, 128, 64], [128, 64], [512, 256, 128, 64]],
              "lr": [0.001, 0.005, 0.0001],
              "dropout": [0, 0.1],
              "activation": ["relu", "leaky_relu"],
              "epochs": [50, 100]
            }


# Create the combinations of the parameters and run training using k-fold cv on grid
trial = 0
best_score = float("inf")
best_config = None

for values in product(param_grid["layers"], param_grid["lr"], param_grid["dropout"], param_grid["activation"], param_grid["epochs"]):
    config = {"layers": values[0], "lr": values[1], "dropout": values[2], "activation": values[3], "epochs": values[4]}

    print(f"\nTrial {trial}")
    print(config)

    results = cross_validation(config, X_train, y_train, 5)
    cv_score = results["cv_mean_loss"]
    print(f"CV mean validation loss: {cv_score}")

    # Save best model for the current grid
    if cv_score < best_score:
        best_score = cv_score
        best_config = config

    trial += 1


print(best_config)
best_model = build_model(best_config)

# Retrain best_model on training
train_val_loss, best_state = train_one_model(best_model, best_config, X_train, y_train, X_train, y_train, loss_fn=nn.MSELoss())

best_model.load_state_dict(best_state)


# Evaluate model performance on training data
best_model.eval()
with torch.no_grad():
    train_predictions = best_model(X_train)
    train_mse = torch.mean((train_predictions - y_train) ** 2)
    train_mae = torch.mean(torch.abs(train_predictions - y_train))


print(f"Train mse: {train_mse.item()}")


# Evaluate model performance on test data
with torch.no_grad():
    test_predictions = best_model(X_test)
    test_mse = torch.mean((test_predictions - y_test) ** 2)
    test_mae = torch.mean(torch.abs(test_predictions - y_test))


print(f"Test mse: {test_mse.item()}")
