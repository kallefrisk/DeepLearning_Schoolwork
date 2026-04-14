import os
import torch
import torch.nn as nn
import mlflow
from itertools import product
from assignment9_functions import split_csvfiles, load, input_target_split, cross_validation, build_model, train_one_model, save_champion_model

print()

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

# print(train_data.columns.tolist())


x_train, y_train = input_target_split(train_data)
# x_val, y_val = input_target_split(val_data)
x_test, y_test = input_target_split(test_data)


# Define paths
model_dir = "MainProject/models"
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
param_grid = {"layers": [[512, 256, 128, 64]],
              "learning_rate": [0.0001],
              "dropout": [0],
              "activation": ["leaky_relu"],
              "optimizer": ["adam"],
              "epochs": [100],
              "patience": [5]
              }


# Create the combinations of the parameters and run training using k-fold cv on grid
trial = 0
best_score = float("inf")
best_config = None

for values in product(param_grid["layers"], param_grid["learning_rate"], param_grid["dropout"], param_grid["activation"], param_grid["optimizer"], param_grid["epochs"], param_grid["patience"]):
    config = {"hidden_layers": values[0], "learning_rate": values[1], "dropout": values[2], "activation": values[3], "optimizer": values[4], "epochs": values[5], "patience": values[6]}

    print(f"\nTrial {trial}")
    print(config)

    results = cross_validation(config, X_train, y_train, 5, device)
    cv_score = results["cv_mean_loss"]
    print(f"CV mean validation loss: {cv_score}")

    # Save best model for the current grid
    if cv_score < best_score:
        best_score = cv_score
        best_config = config

    trial += 1


print(f"\nBest configuration: {best_config}")
best_model = build_model(best_config, device)

# Retrain best_model on training
results = train_one_model(best_model, best_config, X_train, y_train, X_train, y_train)

best_model.load_state_dict(results["best_state"])
save_champion_model(best_model, "champion", results["val_metrics"]["mse"], results["val_metrics"]["mae"], best_config, champion_dir, metadata_dir)


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


print(f"Test mse: {test_mse.item()}\n")
