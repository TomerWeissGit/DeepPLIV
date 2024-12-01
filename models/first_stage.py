import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils.helpers import EarlyStopping, spinner_decorator
import torch.optim as optim


class NeuralNetworkFirstStageWithL1(nn.Module):
    def __init__(self, input_dim: int):
        """
        :param input_dim: int, number of input features.
        """
        super(NeuralNetworkFirstStageWithL1, self).__init__()
        self.input_dim = input_dim
        # Define layers

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim//4+1),
            nn.ReLU(),
            nn.Linear(input_dim//4+1, input_dim//8+1),
            nn.ReLU(),
            nn.Linear(input_dim//8+1, 1))


        # Use the third layer only if the input dimension is greater than 500

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.
        """
        x = self.linear_relu_stack(x)
        return x

    def l1_regularization(self, lambda_l1: float = 0.1) -> torch.Tensor:
        """
        Compute the L1 regularization term.
        """
        l1_reg = torch.tensor(0.0, requires_grad=True)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l1_reg = l1_reg + torch.norm(param, 1)
        return lambda_l1 * l1_reg

    @spinner_decorator("Training first stage")
    def train_new_data(self, x: np.array,
                       y: np.array,
                       epochs: int,
                       learning_rate: float,
                       validation_data: tuple = None,
                       early_stopping_patience: int = 5,
                       early_stopping_min_delta: float = 0.0,
                       print_every_x: int = 100,
                       batch_size: int = 100,
                       num_workers: int = 1) -> None:
        """
        Train the neural network model.
        """

        # Convert numpy arrays to torch tensors
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        x_val_tensor = torch.tensor(validation_data[0], dtype=torch.float32)
        y_val_tensor = torch.tensor(validation_data[1], dtype=torch.float32).view(-1, 1)

        # Define the loss function and the optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

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
            if epoch % print_every_x == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item(): .4f}, Val Loss: {val_loss.item(): .4f}")

            # Check early stopping
            early_stopping(val_loss.item())
            if early_stopping.early_stop:
                print("Early stopping")
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
