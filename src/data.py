import pickle
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
from hydra.utils import to_absolute_path
from pathlib import Path
import hydra
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_digits_subset(config):
    """Load and save a subset of the Digits dataset."""
    logger.info("Loading Digits dataset")
    digits = load_digits()
    X, y = digits.data, digits.target

    logger.info("Splitting dataset")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=config.dataset.train_size,
        test_size=config.dataset.test_size,
        random_state=config.dataset.random_state
    )

    logger.info("Converting to PyTorch tensors")
    X_train = torch.FloatTensor(X_train).reshape(-1, 1, 8, 8)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test).reshape(-1, 1, 8, 8)
    y_test = torch.LongTensor(y_test)

    # Ensure data directory exists
    data_dir = Path(to_absolute_path("data"))
    logger.info(f"Ensuring data directory exists: {data_dir}")
    data_dir.mkdir(exist_ok=True)

    # Save subset to disk
    output_path = to_absolute_path(config.dataset.data_path)
    logger.info(f"Saving dataset to: {output_path}")
    subset = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(subset, f)
        logger.info("Dataset saved successfully")
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")
        raise

    return subset

@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main(cfg):
    logger.info("Starting data preparation")
    load_digits_subset(cfg)
    logger.info("Data preparation completed")

if __name__ == "__main__":
    main()