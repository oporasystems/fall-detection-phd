import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import optuna  # Import Optuna
from training.dataset.fall_detection_dataset import FallDetectionDataset
from training.models.performer import PerformerModel
from training.models.transformer_laurel import TransformerWithLAuReL
from training.utils.dataset_utils import load_dataset
from training.utils.train_utils import train_model, evaluate_model, log_model_size, train_model_optuna
from training.utils.constants import fall_folder, non_fall_folder, max_sequence_length, input_dim, num_classes, \
    csv_columns

from training.utils.logging_utils import create_logger


# Set up logging
logger = create_logger()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

# Constants
BATCH_SIZE = 23

num_heads = 4
num_layers = 1
num_epochs = 20
dropout = 0.006777644883698697
hidden_dim = 8
learning_rate = 0.0007740033761353489

file_paths, labels = load_dataset(fall_folder, non_fall_folder)

# Split data into training, validation, and test sets
train_files, temp_files, train_labels, temp_labels = train_test_split(file_paths, labels, test_size=0.3, random_state=42)
val_files, test_files, val_labels, test_labels = train_test_split(temp_files, temp_labels, test_size=0.5, random_state=42)

# Fit scaler on training data
scaler = StandardScaler()
all_train_data = []
for file in train_files:
    data = pd.read_csv(file)
    acc_data = data[csv_columns].values
    all_train_data.append(acc_data)
all_train_data = np.vstack(all_train_data)
scaler.fit(all_train_data)

# Create PyTorch datasets
train_dataset = FallDetectionDataset(train_files, train_labels)
val_dataset = FallDetectionDataset(val_files, val_labels)
test_dataset = FallDetectionDataset(test_files, test_labels)


hidden_dim_num_heads_map = {
    '2_1': (2, 1), '2_2': (2, 2),
    '4_1': (4, 1), '4_2': (4, 2), '4_4': (4, 4),
    '8_1': (8, 1), '8_2': (8, 2), '8_4': (8, 4), '8_8': (8, 8),
    '12_1': (12, 1), '12_2': (12, 2), '12_4': (12, 4),
    '16_1': (16, 1), '16_2': (16, 2), '16_4': (16, 4), '16_8': (16, 8),
    '24_2': (24, 2), '24_4': (24, 4), '24_8': (24, 8),
    '32_2': (32, 2), '32_4': (32, 4), '32_8': (32, 8), '32_16': (32, 16),
    '40_4': (40, 4), '40_8': (40, 8),
    '48_4': (48, 4), '48_8': (48, 8),
    '64_4': (64, 4), '64_8': (64, 8), '64_16': (64, 16),
    '128_4': (128, 4), '128_8': (128, 8), '128_16': (128, 16)
}


def objective(trial):
    # Suggest hidden_dim and num_heads as a string key, then map it to a tuple
    hidden_dim_num_heads_str = trial.suggest_categorical('hidden_dim_num_heads', list(hidden_dim_num_heads_map.keys()))
    hidden_dim, num_heads = hidden_dim_num_heads_map[hidden_dim_num_heads_str]

    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # Create DataLoaders with the current batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Build the model with hyperparameters
    model = TransformerWithLAuReL(input_dim, num_heads, num_layers, num_classes, hidden_dim, dropout).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    val_acc = train_model_optuna(device, criterion, model, optimizer, train_loader, trial, val_loader)

    return val_acc  # Use validation accuracy as the objective for Optuna


# Run the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1500)

# Retrieve the best hyperparameters
best_params = study.best_trial.params
logger.info(f"Best hyperparameters: {best_params}")

best_hidden_dim, best_num_heads = best_params['hidden_dim_num_heads']

best_num_layers = best_params['num_layers']
best_dropout = best_params['dropout']
best_learning_rate = best_params['learning_rate']
best_batch_size = best_params['batch_size']

# Combine train and validation datasets
combined_train_files = train_files + val_files
combined_train_labels = train_labels + val_labels
combined_train_dataset = FallDetectionDataset(combined_train_files, combined_train_labels, scaler=scaler)

# Create DataLoaders with the best batch size
train_loader = DataLoader(combined_train_dataset, batch_size=best_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_batch_size)

best_model = PerformerModel(input_dim, best_num_heads, best_num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(best_model.parameters(), lr=best_learning_rate)

# Retrain the model with best hyperparameters
num_epochs = 50  # Increase epochs for final training

train_model(device, best_model, train_loader, criterion, optimizer, num_epochs)

# Evaluate the best model
evaluate_model(device, best_model, test_loader)
