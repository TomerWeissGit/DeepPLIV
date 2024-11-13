import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from utils.helpers import EarlyStopping


class NeuralNetworkFirstStage(nn.Module):
    def __init__(self, input_dim: int):
        super(NeuralNetworkFirstStage, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.
        :param x: torch.Tensor, the input tensor.
        :return: torch.Tensor, the output tensor.
        """
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = functional.relu(self.fc3(x))
        x = self.fc4(x)  # Linear activation for the final layer
        return x

    def train_new_data(self, x: np.array, y: np.array, epochs: int, learning_rate: float,
                       early_stopping_patience: int = 30, early_stopping_min_delta: float = 0.0) -> None:
        """
        Train the neural network model.
        :param x: np.array, the input data.
        :param y: np.array, the outcome data.
        :param epochs: int, the number of desired epochs.
        :param learning_rate: float, the learning rate for the optimizer.
        :param early_stopping_patience: int, how many epochs to wait before stopping when loss is not improving.
        :param early_stopping_min_delta: float, minimum change in the monitored quantity to qualify as an improvement.
        """
        # Convert numpy arrays to torch tensors
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        # Define the loss function and the optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Initialize early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta)

        # Training loop
        counter = 0
        for epoch in range(epochs):
            self.train()  # Set the model to training mode

            # Forward pass
            outputs = self(x_tensor)
            loss = criterion(outputs, y_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the loss for every epoch
            if counter % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item(): .4f}')
            counter += 1
            # check early stopping
            early_stopping(loss.item())
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def predict(self, x_new: np.array) -> np.array:
        """
        Predict the outcome for new input data.
        :param x_new: np.array, the new input data.
        :return: np.array, the predicted outcomes.
        """
        # Convert numpy array to torch tensor
        x_new_tensor = torch.tensor(x_new, dtype=torch.float32)

        # Set the model to evaluation mode
        self.eval()

        # Disable gradient computation
        with torch.no_grad():
            # Forward pass
            predictions = self(x_new_tensor)

        # Convert predictions to numpy array and return
        return predictions.numpy()
