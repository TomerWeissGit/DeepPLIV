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
        self.l1_lambda = l1_lambda
        self.act = nn.ReLU()

        # Define layers
        self.fc1 = nn.Linear(input_dim, (input_dim // 4)+1)
        self.bn1 = nn.BatchNorm1d((input_dim // 4)+1)

        self.fc2 = nn.Linear((input_dim // 4)+1, (input_dim // 8)+1)
        self.bn2 = nn.BatchNorm1d((input_dim // 8)+1)

        self.fc3 = nn.Linear((input_dim // 8)+1, (input_dim // 16)+1)
        self.bn3 = nn.BatchNorm1d((input_dim // 16) + 1)


        self.output =nn.Linear((input_dim // 16) + 1, 1) if (input_dim>500) else nn.Linear((input_dim // 8) + 1, 1)
        self.layers = [self.fc1, self.fc2, self.fc3] if input_dim>500 else [self.fc1, self.fc2]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.
        """
        x = self.act(self.bn1(self.fc1(x)))
        x = self.act(self.bn2(self.fc2(x)))
        if self.input_dim > 500:
            x = self.act(self.bn3(self.fc3(x)))
        x = self.output(x)
        return x

    def compute_l1_regularization(self) -> torch.Tensor:
        """
        Compute the L1 regularization for all layers in the network.
        """
        l1_loss = 0.0
        for layer in self.layers:
            for param in layer.parameters():
                l1_loss += torch.norm(param, 1)  # L1 norm of weights
        return self.l1_lambda * l1_loss

    @spinner_decorator("Training first stage")
    def train_new_data(self, x: np.array, y: np.array, epochs: int,
                       learning_rate: float,
                       validation_data: tuple = None,
                       early_stopping_patience: int = 100,
                       early_stopping_min_delta: float = 0.0,
                       print_every_x: int = 100) -> None:
        """
        Train the neural network model.
        """
        if self.l1_lambda is None:
            best_lambda = self._optimize_lambda_with_optuna(x,
                                                            y,
                                                            validation_data,
                                                            epochs,
                                                            learning_rate,
                                                            early_stopping_patience,
                                                            early_stopping_min_delta,
                                                            n_trials=30)
            print(f"Optimal l1_lambda found: {best_lambda}")
            self.l1_lambda = best_lambda
            # Update the model's lambda
        else:
            self.l1_lambda = self.l1_lambda
            # Convert numpy arrays to torch tensors
            x_tensor = torch.tensor(x, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
            x_val_tensor = torch.tensor(validation_data[0], dtype=torch.float32)
            y_val_tensor = torch.tensor(validation_data[1], dtype=torch.float32).view(-1, 1)

            # Define the loss function and the optimizer
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(self.parameters(), lr=learning_rate)

            # Initialize early stopping
            early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta)

            # Training loop
            for epoch in range(epochs):
                optimizer.zero_grad()

                # Forward pass
                outputs = self(x_tensor)
                loss = criterion(outputs, y_tensor)

                # Compute L1 regularization and add to total loss
                l1_loss = self.compute_l1_regularization()
                total_loss = loss + l1_loss

                # Backward pass and optimization
                total_loss.backward()
                optimizer.step()

                # Validation loss
                with torch.no_grad():
                    val_outputs = self(x_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()

                # Print losses
                if epoch % print_every_x == 0:
                    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item(): .4f}, "
                          f"L1 Loss: {l1_loss.item(): .4f}, Val Loss: {val_loss: .4f}")

                # Check early stopping
                early_stopping(loss.item())
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

    def _optimize_lambda_with_optuna(self,
                                     x,
                                     y,
                                     validation_data,
                                     epochs,
                                     learning_rate,
                                     early_stopping_patience,
                                     early_stopping_min_delta,
                                     n_trials):
        """
        Use Optuna to find the best value of l1_lambda.
        """
        def objective(trial):
            lmbda = trial.suggest_loguniform('l1_lambda', 0.01, 0.99)  # Search in log scale
            self.l1_lambda = lmbda  # Temporarily set the lambda for evaluation

            # Prepare data
            x_tensor = torch.tensor(x, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
            x_val_tensor = torch.tensor(validation_data[0], dtype=torch.float32)
            y_val_tensor = torch.tensor(validation_data[1], dtype=torch.float32).view(-1, 1)

            # Define loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(self.parameters(), lr=learning_rate)

            # Train for a subset of epochs
            early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta)
            for epoch in range(epochs):  # Shortened training for lambda optimization
                optimizer.zero_grad()

                outputs = self(x_tensor)
                loss = criterion(outputs, y_tensor)

                l1_loss = self.compute_l1_regularization()
                total_loss = loss + l1_loss

                total_loss.backward()
                optimizer.step()

                # Validation loss
                with torch.no_grad():
                    val_outputs = self(x_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()

                early_stopping(val_loss)
                if early_stopping.early_stop:
                    break
            return val_loss

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        print(f'best lambda found is {study.best_params['l1_lambda']}')
        return study.best_params['l1_lambda']
