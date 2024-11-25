import numpy as np
import torch
import torch.nn as nn
from utils.helpers import EarlyStopping, spinner_decorator
import torch.optim as optim


class NeuralNetworkFirstStage(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.4, weight_decay: float = 0.0, l1_lambda=0):
        super(NeuralNetworkFirstStage, self).__init__()
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda
        self.fc1 = nn.Linear(input_dim, 128)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.AlphaDropout(dropout)

        self.fc2 = nn.Linear(128, 32)
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.AlphaDropout(dropout)

        self.fc3 = nn.Linear(32, 8)
        self.act3 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(8)
        self.dropout3 = nn.AlphaDropout(dropout)

        self.fc4 = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.
        :param x: torch.Tensor, the input tensor.
        :return: torch.Tensor, the output tensor.
        """

        x = self.act1(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.act3(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

    @spinner_decorator("Training first stage")
    def train_new_data(self, x: np.array, y: np.array, epochs: int,
                       learning_rate: float,
                       validation_data: tuple = None,
                       early_stopping_patience: int = 50,
                       early_stopping_min_delta: float = 0.0,
                       print_every_x: int = 200) -> None:
        """
        Train the neural network model.
        :param x: np.array, the input data.
        :param y: np.array, the outcome data.
        :param epochs: int, the number of desired epochs.
        :param learning_rate: float, the learning rate for the optimizer.
        :param validation_data: tuple, the validation data.
        :param early_stopping_patience: int, how many epochs to wait before stopping when loss is not improving.
        :param early_stopping_min_delta: float, minimum change in the monitored quantity to qualify as an improvement.
        :param print_every_x: int, print the loss for every x epochs.
        """
        # Convert numpy arrays to torch tensors
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        x_val_tensor = torch.tensor(validation_data[0], dtype=torch.float32)
        y_val_tensor = torch.tensor(validation_data[1], dtype=torch.float32).view(-1, 1)

        # Define the loss function and the optimizer
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=self.weight_decay)

        # Initialize early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta)

        # Training loop
        counter = 0
        for epoch in range(epochs):
            running_loss = 0.0


            optimizer.zero_grad()

            # Forward pass
            outputs = self(x_tensor)
            loss = criterion(outputs, y_tensor)

            # Add L1 regularization to feature selector weights
            l1_loss = 0.0
            for param in self.fc1.parameters():
                l1_loss += torch.norm(param, 1)  # L1 norm of weights

            total_loss = loss + self.l1_lambda * l1_loss

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            # Print the loss for every epoch
            if validation_data is not None:
                with torch.no_grad():
                    val_outputs = self(x_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                if counter % print_every_x == 0:
                    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item(): .4f}, Val Loss: {val_loss: .4f}')
            else:
                if counter % print_every_x == 0:
                    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item(): .4f}')

            counter += 1
            # check early stopping
            early_stopping(val_loss)
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

