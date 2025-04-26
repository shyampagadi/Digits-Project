import pickle
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
from hydra.utils import to_absolute_path

def load_digits_subset(config):
    """Load and save a subset of the Digits dataset."""
    digits = load_digits()
    X, y = digits.data, digits.target

    # Split into train and test (5% each)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=config.dataset.train_size,
        test_size=config.dataset.test_size,
        random_state=config.dataset.random_state
    )

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).reshape(-1, 1, 8, 8)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test).reshape(-1, 1, 8, 8)
    y_test = torch.LongTensor(y_test)

    # Save subset to disk
    subset = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }
    with open(to_absolute_path(config.dataset.data_path), 'wb') as f:
        pickle.dump(subset, f)

    return subset