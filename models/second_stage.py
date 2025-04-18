import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from utils.helpers import EarlyStopping, spinner_decorator


class NeuralNetworkSecondStage(nn.Module):
    """
    A neural network model for the second stage of the DeepPLIV model.
    """

    def __init__(self, x, v, dropout: float):
        """
        Initialize the neural network model.
        :param x: int, the dimension of the first input.
        :param v: int, the dimension of the second input.
        """
        super(NeuralNetworkSecondStage, self).__init__()

        # First network (deep neural network) for the first input
        self.deep_net = nn.Sequential(

            nn.Linear(x, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Final layer to combine both v and x
        self.final_layer = nn.Linear(v + 16, 1)  # 8 from deep branch and 1 from linear resulting in size 9

    def forward(self, x, v):
        # Pass the first input through the deep neural network
        x1 = self.deep_net(x)

        # Concatenate the output of deep_net with the raw input2
        x = torch.cat((v, x1), dim=1)

        # Pass the concatenated result through the final layer
        output = self.final_layer(x)
        return output

    # @spinner_decorator("Training second stage")
    def train_new_data(self,
                       x_exog: np.array,
                       v_linear: np.array,
                       y: np.array,
                       epochs: int,
                       learning_rate: float,
                       early_stopping_min_delta: float = 0.0,
                       early_stopping_patience: int = None,
                       print_every_x: int = 500,
                       batch_size: int = None) -> None:
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
        :param batch_size: int, the batch size for training.
        """
        # batch size setting
        if not batch_size:
            batch_size = 128 if y.shape[0] < 10000 else 256
        batch_size = batch_size if batch_size else 100
        early_stopping_patience = early_stopping_patience if early_stopping_patience else int(np.sqrt(epochs))
        x_validation = x_exog[:int(x_exog.shape[0] * 0.2)]
        v_linear_validation = v_linear[:int(v_linear.shape[0] * 0.2)]
        y_validation = y[:int(y.shape[0] * 0.2)]
        x_exog_train = x_exog[int(x_exog.shape[0] * 0.2):]
        v_linear_train = v_linear[int(v_linear.shape[0] * 0.2):]
        y_train = y[int(y.shape[0] * 0.2):]
        # Convert numpy arrays to torch tensors
        x_tensor_train = torch.tensor(x_exog_train, dtype=torch.float32)
        v_tensor_train = torch.tensor(v_linear_train, dtype=torch.float32)
        y_tensor_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        x_tensor_validation = torch.tensor(x_validation, dtype=torch.float32)
        v_tensor_validation = torch.tensor(v_linear_validation, dtype=torch.float32)
        y_tensor_validation = torch.tensor(y_validation, dtype=torch.float32).view(-1, 1)

        # Define the loss function and the optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Initialize early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta)
        dataset = TensorDataset(x_tensor_train, v_tensor_train, y_tensor_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        # Training loop
        counter = 0
        self.train()  # Set the model to training mode
        for epoch in range(epochs):
            for x_batch, v_batch, y_batch in dataloader:

                # Forward pass
                outputs = self(x_batch, v_batch)
                loss = criterion(outputs, y_batch)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                val_output = self(x_tensor_validation, v_tensor_validation)
                val_loss = criterion(val_output, y_tensor_validation)

            # Print losses
            # if epoch % print_every_x == 0:
            #     print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item(): .4f}, Val Loss: {val_loss.item(): .4f}")

            # Check early stopping
            early_stopping(val_loss.item())
            if early_stopping.early_stop:
                # print("Early stopping")
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
