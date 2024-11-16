import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.helpers import EarlyStopping, spinner_decorator


class NeuralNetworkSecondStage(nn.Module):
    """
    A neural network model for the second stage of the DeepPLIV model.
    """

    def __init__(self, x, v):
        """
        Initialize the neural network model.
        :param x: int, the dimension of the first input.
        :param v: int, the dimension of the second input.
        """
        super(NeuralNetworkSecondStage, self).__init__()

        # First network (deep neural network) for the first input
        self.deep_net = nn.Sequential(
            nn.Linear(x, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1)

        )

        # Final layer to combine both v and x
        self.final_layer = nn.Linear(1 + v, 1)  # 8 from deep branch and 1 from linear resulting in size 9

    def forward(self, x, v):
        # Pass the first input through the deep neural network
        x1 = self.deep_net(x)

        # Concatenate the output of deep_net with the raw input2
        x = torch.cat((x1, v), dim=1)

        # Pass the concatenated result through the final layer
        output = self.final_layer(x)
        return output

    @spinner_decorator("Training second stage")
    def train_new_data(self,
                       x_exog: np.array,
                       v_linear: np.array,
                       y: np.array,
                       epochs: int,
                       learning_rate: float,
                       early_stopping_min_delta: float = 0.0,
                       early_stopping_patience: int = 100,
                       print_every_x: int = 10000) -> None:
        """
        Train the neural network model.
        :param x_exog: np.array, the input exogenous data.
        :param v_linear: np.array, the input endogenous data.
        :param y: np.array, the outcome data.
        :param epochs: int, the number of desired epochs.
        :param learning_rate: float, the learning rate for the optimizer.
        :param early_stopping_min_delta: float, minimum change in the monitored quantity to qualify as an improvement.
        :param early_stopping_patience: int, how many epochs to wait before stopping when loss is not improving.
        :param print_every_x: int, print the loss every x epochs.
        """
        # Convert numpy arrays to torch tensors
        x_tensor = torch.tensor(x_exog, dtype=torch.float32)
        v_tensor = torch.tensor(v_linear, dtype=torch.float32)
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
            outputs = self(x_tensor, v_tensor)
            loss = criterion(outputs, y_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the loss for every epoch
            if counter % print_every_x == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item(): .4f}')
            counter += 1
            # Check early stopping
            early_stopping(loss.item())
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def predict(self,
                x_new_exog: np.array,
                v_new_endog: np.array) -> np.array:
        """
        Predict the outcome for new input data.
        :param x_new_exog: np.array, the new input data.
        :param v_new_endog: np.array, the new endogenous data.
        :return: np.array, the predicted outcomes.
        """
        # Convert numpy array to torch tensor
        x_new_tensor = torch.tensor(x_new_exog, dtype=torch.float32)
        v_new_tensor = torch.tensor(v_new_endog, dtype=torch.float32)
        # Set the model to evaluation mode
        self.eval()

        # Disable gradient computation
        with torch.no_grad():
            # Forward pass
            predictions = self(x_new_tensor, v_new_tensor)

        # Convert predictions to numpy array and return
        return predictions.numpy()
