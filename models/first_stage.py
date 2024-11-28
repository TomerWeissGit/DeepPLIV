import numpy as np
import torch
import torch.nn as nn
from utils.helpers import EarlyStopping, spinner_decorator
import torch.optim as optim
import optuna


class NeuralNetworkFirstStageWithL1(nn.Module):
    def __init__(self, input_dim: int, l1_lambda: float = 0.0):
        """
        :param input_dim: int, number of input features.
        :param l1_lambda: float, L1 regularization strength for all layers.
        """
        super(NeuralNetworkFirstStageWithL1, self).__init__()
        self.input_dim = input_dim
        self.act = nn.GELU()

        # Define layers
        self.fc1 = nn.Linear(input_dim, 128)

        self.fc2 = nn.Linear(128, 32)

        self.fc3 = nn.Linear(32, 8)


        # Use the third layer only if the input dimension is greater than 500
        self.output =nn.Linear((32, 1) if (input_dim>500) else 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.
        """
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        if self.input_dim > 500:
            x = self.act(self.fc3(x))
        x = self.output(x)
        return x


    @spinner_decorator("Training first stage")
    def train_new_data(self, x: np.array,
                       y: np.array,
                       epochs: int,
                       learning_rate: float,
                       validation_data: tuple = None,
                       early_stopping_patience: int = 800,
                       early_stopping_min_delta: float = 0.0,
                       print_every_x: int = 100) -> None:
        """
        Train the neural network model.
        """
        if self.l1_lambda is None:


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

            # Training loop
            for epoch in range(epochs):
                # optimizer.zero_grad()

                # Forward pass
                outputs = self(x_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                # Validation loss
                val_outputs = self.predict(x_val_tensor)
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
        x_new_tensor = torch.tensor(x_new, dtype=torch.float32)

        # Set the model to evaluation mode
        self.eval()

        # Disable gradient computation
        with torch.no_grad():
            # Forward pass
            predictions = self(x_new_tensor)

        # Convert predictions to numpy array and return
        return predictions.numpy()
