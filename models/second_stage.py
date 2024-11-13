import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


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

        )

        # Final layer to combine both v and x
        self.final_layer = nn.Linear(8 + v, 1)  # 8 from deep branch and 1 from linear resulting in size 9

    def forward(self, x, v):
        # Pass the first input through the deep neural network
        x1 = self.deep_net(x)

        # Concatenate the output of deep_net with the raw input2
        x = torch.cat((x1, v), dim=1)

        # Pass the concatenated result through the final layer
        output = self.final_layer(x)
        return output



    def train_new_data(self,
                       x_exog: np.array,
                       v_linear: np.array,
                       y: np.array,
                       epochs: int,
                       learning_rate: float) -> None:
        """
        Train the neural network model.
        :param x_exog: np.array, the input exogenous data.
        :param v_linear: np.array, the input endogenous data.
        :param y: np.array, the outcome data.
        :param epochs: int, the number of desired epochs.
        :param learning_rate: float, the learning rate for the optimizer.
        """
        # Convert numpy arrays to torch tensors
        x_tensor = torch.tensor(x_exog, dtype=torch.float32)
        v_tensor = torch.tensor(v_linear, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        # Define the loss function and the optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Training loop
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
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item(): .4f}')

    def predict(self, x_new_exog: np.array, v_new_endog: np.array) -> np.array:
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
