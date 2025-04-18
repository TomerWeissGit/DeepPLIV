import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils.helpers import EarlyStopping, spinner_decorator
import torch.optim as optim


class NeuralNetworkFirstStage(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, dropout: float = 0.6):
        """
        :param input_dim: int, number of input features.
        """
        super(NeuralNetworkFirstStage, self).__init__()
        self.input_dim = input_dim
        # Define layers

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, output_dim))



        # Use the third layer only if the input dimension is greater than 500

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.
        """
        x = self.linear_relu_stack(x)
        return x


    # @spinner_decorator("Training first stage")
    def train_new_data(self, x: np.array,
                       y: np.array,
                       epochs: int,
                       learning_rate: float,
                       validation_data: tuple = None,
                       early_stopping_patience: int = None,
                       early_stopping_min_delta: float = 0.0,
                       print_every_x: int = 150) -> None:
        """
        Train the neural network model.
        """
        batch_size = 128 if y.shape[0] < 10000 else 256
        early_stopping_patience = early_stopping_patience if early_stopping_patience else int(np.sqrt(epochs))

        # Convert numpy arrays to torch tensors
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        x_val_tensor = torch.tensor(validation_data[0], dtype=torch.float32)
        y_val_tensor = torch.tensor(validation_data[1], dtype=torch.float32).view(-1, 1)

        # Define the loss function and the optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.001, betas=(0.9, 0.999), eps=1e-08)

        # Initialize early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta)

        # Create dataset and DataLoader
        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        # Training loop
        for epoch in range(epochs):
            self.train()  # Set the model to training mode

            for x_batch, y_batch in dataloader:

                # Forward pass
                outputs = self(x_batch)
                loss = criterion(outputs, y_batch)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation loop (run after each epoch)
            self.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                val_outputs = self(x_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)

            # Print losses
            # if epoch % print_every_x == 0:
                # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item(): .4f}, Val Loss: {val_loss.item(): .4f}")

            # Check early stopping
            early_stopping(val_loss.item())
            if early_stopping.early_stop:
                # print("Early stopping")
                break

    def predict(self, x_new: np.array) -> np.array:
        """
        Predict the outcome for new input data.
        """
        # Convert numpy array to torch tensor
        x_new_tensor = torch.tensor(x_new, dtype=torch.float32).clone().detach().requires_grad_(False)

        # Set the model to evaluation mode
        self.eval()

        # Disable gradient computation
        with torch.no_grad():
            # Forward pass
            predictions = self(x_new_tensor)

        # Convert predictions to numpy array and return
        return predictions.numpy()
