import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MLPBinaryPredictor(nn.Module):
    """
    Simple MLP for binary classification using ESM-2 protein embeddings
    """
    def __init__(self, input_size, hidden_size, output_size = 1, layer_count=3, dropout=0.3):
        super(MLPBinaryPredictor, self).__init__()
        self.relu = nn.ReLU()

        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        # Hidden layers
        for _ in range(layer_count - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))

        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.model(x)
        
        return out

# MPL Wrapper for sklearn compatibility
class SimpleMLP(nn.Module):
    def __init__(self, epochs=10, batch_size=32, learning_rate=1e-3, input_size=1280, hidden_size=512, dropout=0.3, layer_count=3):
        super(SimpleMLP, self).__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = MLPBinaryPredictor(input_size=input_size, hidden_size=hidden_size, dropout=dropout, layer_count=layer_count)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def fit(self, X, y):
        self.model.train()
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
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
