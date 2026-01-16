import optuna
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from training.utils.logging_utils import create_logger

logger = create_logger()


def train_model_optuna(device, criterion, model, optimizer, train_loader, trial, val_loader):
    # Training loop
    num_epochs = 10  # Fewer epochs for hyperparameter optimization
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Evaluate on validation set
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Calculate validation accuracy
        val_acc = accuracy_score(val_labels, val_preds)

        # Log the training and validation results
        logger.info(f'Trial {trial.number}, Epoch {epoch + 1}/{num_epochs}, '
                    f'Train Loss: {running_loss:.4f}, Val Accuracy: {val_acc * 100:.2f}%')

        # Report intermediate objective value (validation accuracy) to Optuna
        trial.report(val_acc, epoch)

        # Handle pruning
        if trial.should_prune():
            logger.info(f'Trial {trial.number} pruned at epoch {epoch + 1}')
            raise optuna.exceptions.TrialPruned()
    # Log the model size after training
    model_size_in_mb = log_model_size(model)
    logger.info(f"Model size after training: {model_size_in_mb:.2f} MB")
    # If model size exceeds 5 MB, prune the trial after training
    if model_size_in_mb > 5:
        logger.info(
            f"Pruning trial {trial.number} because model size exceeds 5 MB after training ({model_size_in_mb:.2f} MB)")
        raise optuna.exceptions.TrialPruned()
    return val_acc

# Train model function
def train_model(device, model, train_loader, test_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, Accuracy: {train_acc:.2f}%')

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_acc = 100 * correct / total
        print(f'Test Accuracy: {test_acc:.2f}%')


# Evaluate the model
def evaluate_model(device, model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Non-Fall', 'Fall']))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Fall', 'Fall'],
                yticklabels=['Non-Fall', 'Fall'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def log_model_size(model):
    # Calculate the total number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters())

    # Assume each parameter is a float32 (4 bytes)
    param_size_in_bytes = num_params * 4

    # Convert bytes to megabytes
    model_size_in_mb = param_size_in_bytes / (1024 * 1024)

    return model_size_in_mb