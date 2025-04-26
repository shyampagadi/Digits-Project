import torch
import torch.nn as nn
import torch.optim as optim
import hydra
from hydra.utils import to_absolute_path
from pathlib import Path
from src.data import load_digits_subset
from src.model import SimpleMLP
from src.utils import load_subset

@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main(cfg):
    # Set random seed
    torch.manual_seed(cfg.dataset.random_state)

    # Ensure data directory exists
    Path(to_absolute_path("data")).mkdir(exist_ok=True)

    # Load or create dataset
    try:
        subset = load_subset(to_absolute_path(cfg.dataset.data_path))
    except FileNotFoundError:
        subset = load_digits_subset(cfg)

    X_train, y_train = subset['X_train'], subset['y_train']
    X_test, y_test = subset['X_test'], subset['y_test']

    # Initialize model
    model = SimpleMLP(cfg.model.hidden_units, cfg.model.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.training.learning_rate)

    # Training loop
    for epoch in range(cfg.training.epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).float().mean().item()
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    # Ensure models directory exists
    Path(to_absolute_path("models")).mkdir(exist_ok=True)

    # Save model
    torch.save(model.state_dict(), to_absolute_path(cfg.model.model_path))

if __name__ == "__main__":
    main()