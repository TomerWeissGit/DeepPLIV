"""A module for a mixture density network layer

For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader

from utils.helpers import EarlyStopping, spinner_decorator


class MDN(nn.Module):
    """A mixture density network layer

    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.

    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions

    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.

    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, minibatch):
        pi = self.pi(minibatch)
        sigma = torch.exp(self.sigma(minibatch))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


def gaussian_probability(sigma, mu, target):
    """Returns the probability of `target` given MoG parameters `sigma` and `mu`.

    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        target (BxI): A batch of target. B is the batch size and I is the number of
            input dimensions.

    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    target = target.unsqueeze(1).expand_as(sigma)
    ret = 1.0 / math.sqrt(2 * math.pi) * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
    return torch.prod(ret, 2)


class NeuralNetworkFirstStageMDN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, dropout: float = 0.0, num_gausians = 5):
        """
        :param input_dim: int, number of input features.
        """
        super(NeuralNetworkFirstStageMDN, self).__init__()
        self.input_dim = input_dim
        # Define layers

        self.linear_relu_stack = nn.Sequential(nn.Linear(input_dim, 128),
                                               nn.ReLU(),
                                               nn.Dropout(dropout),
                                               nn.Linear(128, 64),
                                               nn.Dropout(dropout),
                                               nn.ReLU(),
                                               nn.Linear(64, 32),
                                               nn.ReLU(),
                                               nn.Dropout(dropout),
                                               MDN(32, output_dim, num_gaussians=num_gausians))


        # Use the third layer only if the input dimension is greater than 500

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.
        """
        x = self.linear_relu_stack(x)
        return x


    @spinner_decorator("Training first stage")
    def train_new_data(self, x: np.array,
                       y: np.array,
                       epochs: int,
                       learning_rate: float,
                       validation_data: tuple = None,
                       early_stopping_patience: int = None,
                       early_stopping_min_delta: float = 0.0,
                       print_every_x: int = 10,
                       batch_size: int = None) -> None:
        """
        Train the neural network model.
        """
        # Convert numpy arrays to torch tensors
        if not batch_size:
            batch_size = 128 if y.shape[0] < 10000 else 256
        batch_size = batch_size if batch_size else 100
        early_stopping_patience = early_stopping_patience if early_stopping_patience else int(np.sqrt(epochs))
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        x_val_tensor = torch.tensor(validation_data[0], dtype=torch.float32)
        y_val_tensor = torch.tensor(validation_data[1], dtype=torch.float32).view(-1, 1)

        # Define the loss function and the optimizer
        self.zero_grad()
        pi, sigma, mu = self(x_tensor)
        loss = self._mdn_loss(pi, sigma, mu, y_tensor)

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
                pi, sigma, mu = self(x_batch)
                loss = self._mdn_loss(pi, sigma, mu, y_batch)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation loop (run after each epoch)
            self.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                val_pi, val_sigma, val_mu = self(x_val_tensor)
                val_loss = self._mdn_loss(val_pi, val_sigma, val_mu, y_val_tensor)

            # Print losses
            if epoch % print_every_x == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item(): .4f}, Val Loss: {val_loss.item(): .4f}")

            # Check early stopping
            early_stopping(val_loss.item())
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def predict_mean(self, x_new: np.array, num_samples: int = 50) -> np.array:
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
            pi, sigma, mu = self(x_new_tensor)
            samples = np.array([np.array(self._sample(pi, sigma, mu)) for x in range(num_samples)])
        # Convert predictions to numpy array and return
        return samples.reshape(num_samples, samples.shape[1]).mean(axis=0)


    @staticmethod
    def _mdn_loss(pi, sigma, mu, target):
        """Calculates the error, given the MoG parameters and the target

        The loss is the negative log likelihood of the data given the MoG
        parameters.
        """
        prob = pi * gaussian_probability(sigma, mu, target)
        nll = -torch.log(torch.sum(prob, dim=1) + 1e-9)
        return torch.mean(nll)

    @staticmethod
    def _sample(pi, sigma, mu):
        """Draw samples from a MoG.
        """
        # Choose which gaussian we'll sample from
        pis = Categorical(pi).sample().view(pi.size(0), 1, 1)
        # Choose a random sample, one randn for batch X output dims
        # Do a (output dims)X(batch size) tensor here, so the broadcast works in
        # the next step, but we have to transpose back.
        gaussian_noise = torch.randn(
            (sigma.size(2), sigma.size(0)), requires_grad=False)
        variance_samples = sigma.gather(1, pis).detach().squeeze()
        mean_samples = mu.detach().gather(1, pis).squeeze()
        return (gaussian_noise * variance_samples + mean_samples).transpose(0, 1)


