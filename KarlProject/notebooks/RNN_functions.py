import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import copy


def train_model(params: dict, train_data: torch.Tensor, train_labels: torch.Tensor, device, n_folds: int = 5, classifier: bool = True) -> tuple[nn.Module, dict]:
    """
    Train model with cross-validation

    Args:
        params: Training parameters
        train_data: tesor of X data
        train_labels: tensor of Y data
        device: the device to send the models to
        n_folds: Number of folds for cross-validation (default: 5)
    """

    if classifier:
        # Compute weights of the labels to counteract an imbalanced dataset
        n_positive = 0
        n_negative = 0
        for Y in train_labels:
            if Y == 1:
                n_positive += 1
            else:
                n_negative += 1

        pos_weight = torch.tensor([n_negative / n_positive])
    else:
        pos_weight = 0

    results = {}

    # Perform k-fold cross-validation
    print(f"\nPerforming {n_folds}-fold cross-validation...")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        # Split data
        train_x_fold = train_data[train_idx]
        train_y_fold = train_labels[train_idx]
        val_x_fold = train_data[val_idx]
        val_y_fold = train_labels[val_idx]

        # Train on this fold
        model, fold_stats = train_single_model(
            params, train_x_fold, train_y_fold, device, val_x_fold,
            val_y_fold, pos_weight, fold_num=fold + 1, classifier=classifier)

        fold_results.append(fold_stats)
        fold_models.append(model)

    # Aggregate cross-validation results
    results = aggregate_cv_results(fold_results)
    results['fold_models'] = fold_models
    results['cv_scores'] = [stats['final_val_accuracy'] for stats in fold_results]
    results['mean_cv_accuracy'] = np.mean(results['cv_scores'])
    results['std_cv_accuracy'] = np.std(results['cv_scores'])

    # Train final model on all data
    print("\n--- Training final model on all data ---")
    final_model, final_stats = train_single_model(
        params, train_data, train_labels, device,
        pos_weight, is_final=True, classifier=classifier)

    results['final_model'] = final_model
    results['final_stats'] = final_stats

    return final_model, results


def train_single_model(params: dict, train_data: torch.Tensor, train_labels: torch.Tensor, device,
                       val_data: torch.tensor = None, val_labels: torch.tensor = None,
                       pos_weight: torch.Tensor = None, fold_num: int = None,
                       is_final: bool = False, classifier: bool = True) -> tuple[nn.Module, dict]:
    """Train a single model with validation"""

    model = params["model_type"](
        params["input_size"],
        params["hidden_size"],
        params["depth"],
        params["num_of_classification_labels"]
    ).to(device)

    optimizer = params["optimizer"](model.parameters(), lr=params["lr"])
    if classifier:
        loss_func = params["loss_func"](pos_weight=pos_weight)
    else:
        loss_func = params["loss_func"]()

    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)

    if val_data is not None and val_labels is not None:
        val_dataset = TensorDataset(val_data, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, drop_last=False)

    prev_loss = np.inf
    streak = 0
    loss_increase = False
    loss_history = []
    val_loss_history = []
    val_accuracy_history = []
    early_stopping = False

    prefix = f"Fold {fold_num}" if fold_num else "Final" if is_final else "Model"
    print(f"{prefix} training with {params['epochs']} epochs")
    print("Start", " " * 10, "Finished")
    print("v", " " * 21, "v")

    interval = max(1, params["epochs"] // 25)

    for epoch in range(params["epochs"]):
        # Training phase
        model.train()
        total_loss = 0

        for X, Y in train_loader:

            output = model(X)
            loss = loss_func(output, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        loss_history.append(avg_loss)

        # Validation phase
        if val_data is not None and val_labels is not None:
            model.eval()
            val_loss = 0
            correct = 0
            total_samples = 0

            with torch.no_grad():
                for X, Y in val_loader:
                    output = model(X)
                    loss = loss_func(output, Y)
                    val_loss += loss.item()

                    # Calculate accuracy
                    _, predicted = torch.max(output, 1)
                    correct += (predicted == Y).sum().item()
                    total_samples += 1

            avg_val_loss = val_loss / len(val_data)
            val_accuracy = correct / total_samples
            val_loss_history.append(avg_val_loss)
            val_accuracy_history.append(val_accuracy)

            # Early stopping based on validation loss
            if avg_val_loss > prev_loss:
                loss_increase = True
            else:
                loss_increase = False
                # Save best model based on validation loss
                if not is_final:
                    best_model_state = copy.deepcopy(model.state_dict())

            if loss_increase:
                streak += 1
            else:
                streak = 0

            prev_loss = avg_val_loss
        else:
            # Early stopping based on training loss
            if avg_loss > prev_loss:
                loss_increase = True
            else:
                loss_increase = False

            if loss_increase:
                streak += 1
            else:
                streak = 0

            prev_loss = avg_loss

        # Early stopping check
        if streak >= params.get("patience", 5):
            early_stopping = True
            print("X")
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

        # Progress indicator
        if epoch % interval == 0:
            print("#", end="", flush=True)

    print()

    # Load best model
    if val_data is not None and val_labels is not None and not is_final and 'best_model_state' in locals():
        model.load_state_dict(best_model_state)

    # Calculate final validation metrics
    final_val_accuracy = val_accuracy_history[-1] if val_accuracy_history else None
    best_val_accuracy = max(val_accuracy_history) if val_accuracy_history else None

    stats = {
        "loss_history": loss_history,
        "val_loss_history": val_loss_history,
        "val_accuracy_history": val_accuracy_history,
        "final_val_accuracy": final_val_accuracy,
        "best_val_accuracy": best_val_accuracy,
        "early_stopping": early_stopping,
        "n_epochs_trained": len(loss_history),
        "optimizer": optimizer
    }

    return model, stats


def aggregate_cv_results(fold_results: list) -> dict:
    """Aggregate results from cross-validation folds"""

    aggregated = {
        "cv_fold_results": fold_results,
        "val_accuracies": [r['final_val_accuracy'] for r in fold_results if r['final_val_accuracy']],
        "best_val_accuracies": [r['best_val_accuracy'] for r in fold_results if r['best_val_accuracy']],
        "n_epochs_per_fold": [r['n_epochs_trained'] for r in fold_results]
    }

    if aggregated['val_accuracies']:
        aggregated['mean_val_accuracy'] = np.mean(aggregated['val_accuracies'])
        aggregated['std_val_accuracy'] = np.std(aggregated['val_accuracies'])

    if aggregated['best_val_accuracies']:
        aggregated['mean_best_accuracy'] = np.mean(aggregated['best_val_accuracies'])

    print("\n" + "="*50)
    print("Cross-Validation Results:")
    if aggregated['val_accuracies'] and aggregated['std_val_accuracy']:
        print(f"  Mean validation accuracy: {aggregated['mean_val_accuracy']:.4f} ± {aggregated['std_val_accuracy']:.4f}")
    if aggregated['best_val_accuracies']:
        print(f"  Mean best accuracy: {aggregated['mean_best_accuracy']:.4f}")
    print(f"  Mean epochs trained: {np.mean(aggregated['n_epochs_per_fold']):.1f}")
    print("="*50)

    return aggregated
