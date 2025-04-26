import pickle

def load_subset(data_path):
    """Load dataset subset from disk."""
    with open(data_path, 'rb') as f:
        subset = pickle.load(f)
    return subset