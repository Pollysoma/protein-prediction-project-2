import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Any, Dict


class MLPBinaryPredictor(nn.Module):
    """Simple MLP for binary classification using protein embeddings."""

    def __init__(self, input_size, hidden_size, output_size=1, layer_count=3, dropout=0.3):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(layer_count - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class SimpleMLP(nn.Module):
    """Sklearn-like wrapper around the MLP for quick baselines."""

    def __init__(
        self,
        epochs=10,
        batch_size=32,
        learning_rate=1e-3,
        input_size=1280,
        hidden_size=512,
        dropout=0.3,
        layer_count=3,
    ):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.layer_count = layer_count
        self.model = MLPBinaryPredictor(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            layer_count=self.layer_count,
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "layer_count": self.layer_count,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        # rebuild model/optimizer with updated hyperparameters
        self.model = MLPBinaryPredictor(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            layer_count=self.layer_count,
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return self

    def fit(self, X, y):
        self.model.train()
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.epochs):
            for inputs, labels in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(inputs).squeeze()
            probs = torch.sigmoid(outputs).numpy()
        return np.vstack([1 - probs, probs]).T
