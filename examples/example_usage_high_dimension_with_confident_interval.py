
import itertools
import random

import statsmodels.api as sm

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        """
        Early stopping to stop the training when the loss does not improve after certain epochs.
        :param patience: int, how many epochs to wait before stopping when loss is not improving.
        :param min_delta: float, minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def save_to_pickle(lst: Iterable, name: str = 'results.pkl'):
    import pickle
    with open(name, 'wb') as f:
        pickle.dump(lst, f)

def plot_boxplot(df, y_line: float = None):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='coefficient', y='value', hue='method', data=df)
    if y_line:
        plt.hlines(y=y_line, xmin=-1, xmax=4, colors='r', linestyles='--', lw=2)
    plt.title('Coefficient Distribution by Method')
    plt.show()



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

    def train_new_data(self,
                       x_exog: np.array,
                       v_linear: np.array,
                       y: np.array,
                       epochs: int,
                       learning_rate: float,
                       early_stopping_min_delta: float = 0.0,
                       early_stopping_patience: int = None,
                       print_every_x: int = 150) -> None:
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
        batch_size = 256
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
            if epoch % print_every_x == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item(): .4f}, Val Loss: {val_loss.item(): .4f}")

            # Check early stopping
            early_stopping(val_loss.item())
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
        batch_size =  256
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


class DeepPLIV:
    def __init__(self):
        self.first_stage_model: NeuralNetworkFirstStage = None
        self.second_stage_model: NeuralNetworkSecondStage = None
        pass

    def fit(self, v_1: np.array, z_1: np.array,z_2: np.array, x: np.array, y: np.array,
            first_stage_epochs: int = 500,
            first_stage_learning_rate: float = 0.01,
            dropout: float = 0.6,
            second_stage_epochs: int = 500,
            second_stage_learning_rate: float = 0.01) -> (NeuralNetworkFirstStage, NeuralNetworkSecondStage):
        """
        Fit the DeepPLIV model. The model is trained in two stages:
            - First stage: Train a neural network to predict the endogenous variable.
            - Second stage: Train a neural network to predict the outcome variable.
        :param v_1: the dependent variable.
        :param z_1: the instrumental variable. for the first stage.
        :param x: the exogenous variable.
        :param z_2: the instrumental variable for the second stage.
        :param y: the outcome variable.
        :param first_stage_epochs: the number of epochs for training the first stage.
        :param first_stage_learning_rate: the learning rate for training the first stage.
        :param dropout: the dropout rate for the first stage.
        :param second_stage_epochs: the number of epochs for training the second stage.
        :param second_stage_learning_rate: the learning rate for training the second stage.
        :return:
        """
        self.fit_first_stage(v_1, z_1, first_stage_epochs, first_stage_learning_rate, dropout=dropout)
        v_hat = self.predict_first_stage(z_2)
        self.fit_second_stage(v_hat, x, y, second_stage_epochs, second_stage_learning_rate, dropout=dropout)
        return self.first_stage_model, self.second_stage_model

    def predict(self,
                z: np.array,
                x: np.array) -> np.array:
        """
        Predict the outcome variable using the DeepPLIV model.
        :param z: the instrumental variable for the second stage.
        :param x: the exogenous variable.
        :return: the predicted outcome variable.
        """
        v_hat = self.predict_first_stage(z)
        return self.predict_second_stage(v_hat, x)

    def fit_first_stage(self, v_1: np.array, z_1: np.array,
                        epochs_first_stage: int,
                        learning_rate_first_stage: float ,
                        dropout: float = 0,
                        validation_data: tuple = None,
                        output_dim = 1) -> NeuralNetworkFirstStage:
        """
        Fit the DeepPLIV model.
        :param v_1: np.array, the dependent variable.
        :param z_1: np.array, the instrumental variable.
        :param epochs_first_stage: int, the number of epochs for training the first stage.
        :param learning_rate_first_stage: float, the learning rate for training the first stage.
        :param l1_lambda: float, the L1 regularization parameter.
        :param validation_data: tuple, the validation data.
        :return: NeuralNetworkFirstStage, the trained first stage model.
        """

        self.first_stage_model = NeuralNetworkFirstStage(input_dim=z_1.shape[1], output_dim=output_dim,
                                                         dropout=dropout)
        self.first_stage_model.train_new_data(z_1, v_1, epochs_first_stage, learning_rate_first_stage,
                                              validation_data=validation_data)
        return self.first_stage_model

    def fit_second_stage(self, v_hat: np.array, x: np.array, y: np.array,
                         epochs_second_stage: int,
                         learning_rate_second_stage: float,
                         dropout: float) -> NeuralNetworkSecondStage:
        """
        Fit the second stage of the DeepPLIV model.
        :param v_hat: np.array, the dependent variable prediction.
        :param x: np.array, the exogenous variable.
        :param y: np.array, the outcome variable.
        :param dropout: float, the dropout rate for the second stage.
        :param epochs_second_stage: int, the number of epochs for training the second stage.
        :param learning_rate_second_stage: float, the learning rate for training the second stage.
        """
        self.second_stage_model = NeuralNetworkSecondStage(x=x.shape[1], v=v_hat.shape[1], dropout=dropout)
        self.second_stage_model.train_new_data(x_exog=x,
                                               v_linear=v_hat,
                                               y=y,
                                               epochs=epochs_second_stage,
                                               learning_rate=learning_rate_second_stage)
        return self.second_stage_model

    def predict_first_stage(self, z_2: np.array) -> np.array:
        """
        Predict the endogenous variable.
        :param z_2: np.array, the instrumental variable.
        :return: np.array, the predicted endogenous variable.
        """
        return self.first_stage_model.predict(z_2)

    def predict_second_stage(self, v_hat: np.array, x: np.array) -> np.array:
        """
        Predict the outcome.
        :param v_hat: np.array, the predicted endogenous variable.
        :param x: np.array, the exogenous variable.
        :return: np.array, the predicted outcome.
        """
        return self.second_stage_model.predict(v_new_endog=v_hat, x_new_exog=x)

    def get_v_predicted_coefficient(self) -> float:
        """
        Get the coefficient of v_predicted from the second stage model.
        :return: float, the coefficient of v_predicted.
        """
        # Assuming the first layer of the second stage model is a linear layer
        return self.second_stage_model.final_layer.weight[0, -1].item()
    pass

class SimDataCreatorHighDimension:
    """
    this class replicates the logic of 'generate_data_from_regression' in an oop style,
    similar to 'sim_data_creator_high_dimension'.

    attributes
    ----------
    n_x : int
        sample size for the first stage (to estimate gamma_iv).
    n_y : int
        sample size for the second stage (to generate final outcome y).
    m : int
        number of snps to start with.
    betas : tuple
        (beta0, beta1, beta2, beta_u) for the y model.
    gamma_u : tuple
        gamma_u for the x model.
    p_thr : float
        p-value threshold for snp filtering.

    after instantiation, call get_data() to retrieve data arrays:
        x_1, x_2, y, z_1, z_2
    """

    def __init__(self, n, m, betas, gamma_u, p_thr, interaction = False):
        self.n_x = int(n//2)
        self.n_y = int(n//2)
        self.m = int(m)
        self.betas = betas  # (beta0, beta1, beta2, beta_u)
        self.gamma_u = gamma_u  # (gamma_0, gamma_u)
        self.p_thr = p_thr

        # generate all data upon instantiation
        self.x_1, self.x_2, self.z_1, self.z_2, self.g, self.y = self._generate_data(interaction)

    def _generate_data(self, interaction):
        """
        main data-generation pipeline. loosely follows the structure of
        'generate_data_from_regression', divided into:
          1) first stage (n_x) for x
          2) second stage (n_y) for y
        """

        # unpack parameters
        beta1, beta2, beta_u = self.betas
        gamma_u = self.gamma_u

        # first-stage generation (n_x)
        # create m snp effects from n(0, sqrt(5e-3)) distribution
        sigma_iv = np.sqrt(5e-3)
        gamma_ivs = np.random.normal(0, sigma_iv, self.m)

        # create "gwas" estimates for each snp ~ normal(gamma_iv, sqrt(1/n_x))
        gamma_ivs_for_gwas = np.random.normal(loc=gamma_ivs, scale=np.sqrt(1 / self.n_x), size=self.m)

        # filter snps by p-value < p_thr
        p_values = 2 * (1 - norm.cdf(np.sqrt(self.n_x) * np.abs(gamma_ivs_for_gwas)))
        mask = (p_values < self.p_thr)
        gamma_ivs_filtered = gamma_ivs[mask][:500] if len(gamma_ivs[mask]) > 500 else gamma_ivs[mask]
        number_of_ivs = len(gamma_ivs_filtered)

        # generate x for the n_x individuals using filtered snps
        g_iv_n_x = np.random.binomial(n=2, p=0.3, size=(self.n_x, number_of_ivs))
        g_iv_n_y = np.random.binomial(n=2, p=0.3, size=(self.n_y, number_of_ivs))

        # unobserved confounder
        u_n_x = np.random.normal(0, 1, self.n_x)
        u_n_y = np.random.normal(0, 1, self.n_y)

        # x ~  sum_j(gamma_ivs_filtered_j * g_iv_n_x_j) + gamma_u*u + normal(0,1)
        if interaction:
            n_pairs = g_iv_n_x.shape[1]//4
            # number of pairs you want to use
            # (here we just do k//2, but you could do something else)
            # all possible pairs of indices (i < j)
            all_pairs = list(itertools.combinations(range(n_pairs), 2))
            # randomly select n_pairs of them
            chosen_pairs = random.sample(all_pairs, n_pairs)

            x_n_x = self.generate_x_with_interactions(g_iv=g_iv_n_x, gamma_ivs_filtered=gamma_ivs_filtered,
                                                      chosen_pairs=chosen_pairs) + gamma_u * u_n_x
            x_n_y = self.generate_x_with_interactions(g_iv=g_iv_n_y, gamma_ivs_filtered=gamma_ivs_filtered,
                                                      chosen_pairs=chosen_pairs) + gamma_u * u_n_y
        else:
            x_n_x = g_iv_n_x @ gamma_ivs_filtered + gamma_u * u_n_x
            x_n_y = g_iv_n_y @ gamma_ivs_filtered + gamma_u * u_n_y
        # 2) second-stage generation (n_y)
        # create a single snp g that interacts with x
        g = np.random.binomial(2, 0.25, self.n_y)
        # mean-center g
        g = g - np.mean(g)
        if interaction:
            scaler = StandardScaler()
            y = (beta1 * x_n_y + (np.exp(-beta2 * g + 5)) ** 1.2 * ((g ** 2) ** 0.5) + beta_u * u_n_y
                 + np.random.normal(0, 1, self.n_y))
        else:
            y = beta1 * x_n_y + beta2 * g  + beta_u * u_n_y + np.random.normal(0, 1, self.n_y)

        # for demonstration, let z_2 be the set of original gamma_ivs (pre-filter)
        return x_n_x, x_n_y, g_iv_n_x, g_iv_n_y, g, y

    def get_data(self) -> tuple:
        """
        return data in a structure similar to the original class:
            x_1, x_2, z_1, z_2, g, y
        """
        return (
            self.x_1,
            self.x_2,
            self.z_1,
            self.z_2,
            self.g,
            self.y
        )
    @staticmethod
    def generate_x_with_interactions(
            g_iv: np.ndarray,
            gamma_ivs_filtered: np.ndarray,
            chosen_pairs: np.array,
    ) -> np.ndarray:
        """
        generate x from interactions among random pairs of ivs,
        plus a confounder gamma_u * u_n_x, plus normal(0, 1) noise.

        parameters
        ----------
        g_iv : np.ndarray
            (n_x, k) genotype matrix after mean-centering,
            where k = len(gamma_ivs_filtered).
        gamma_ivs_filtered : np.ndarray
            the array of iv effects (gamma_i), length = k.
        chosen_pairs : np.array
        returns
        -------
        x : np.ndarray
            shape (n_x,). x generated from pairwise snp interactions,
            confounder, and normal(0,1) noise.
        """

        # initialize x contribution from interactions
        x_contrib = np.zeros(g_iv.shape[0])

        # accumulate pairwise interaction effects
        for (i, j) in chosen_pairs:
            interaction_effect = ((np.abs(gamma_ivs_filtered[i]) + np.abs(gamma_ivs_filtered[j]))
                                  * (np.sign(gamma_ivs_filtered[i])) * np.sign(gamma_ivs_filtered[j]))
            x_contrib += interaction_effect * (g_iv[:, i] * g_iv[:, j])

        # add confounder and random noise
        x = x_contrib

        return x


class NaiveSRISPSHighDimension:
    def __init__(self,
                 data_creator: SimDataCreatorHighDimension,
                 epochs: int = 1000,
                 learning_rate: float = 0.001,
                 dropout: float = 0,
                 b: int = 30,
                 m: int = 10):
        """
        Initialize the NaiveSRISPSHighDimension class.
        :param data_creator: SimDataCreatorHighDimension, an instance of the SimDataCreatorHighDimension class.
        :param epochs: int, the number of epochs for training the neural network model.
        :param learning_rate: float, the learning rate for training the neural network model.
        """
        self.data_creator = data_creator
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.x_1, self.x_2, self.z_1, self.z_2, self.g, self.y = data_creator.get_data()
        self.dropout = dropout
        self.xa = self.z_1
        self.xb = self.z_2
        self.b = b
        self.m = m

    def estimate_naive_regression(self):
        """
        Estimate the Naive SR-IV model using a simple linear regression model.
        :return: np.array, the coefficients of the Naive SR-IV model.
        """
        # 1) Organize X: add a constant column for intercept
        x = np.column_stack([
            self.x_2,
            self.g,
        ])
        x = sm.add_constant(x)  # add intercept

        # 2) Fit OLS
        model_sm = sm.OLS(self.y, x).fit()
        param = model_sm.params[1]  # pandas Series of parameter estimates
        conf_95 = model_sm.conf_int(0.05)[1]  # DataFrame with 2 columns [lower, upper]
        conf_90 = model_sm.conf_int(0.1)[1] # DataFrame with 2 columns [lower, upper]
        return param, conf_95, conf_90

    def estimate_sps(self):
        """
        Estimate the 2SLS model using a simple linear regression model. - SPS
        :return: np.array, the coefficients of the Naive SR-IV model.
        """
        model = LinearRegression()

        model.fit(self.xa, self.x_1)
        x_predicted = model.predict(self.xb)

        # 1) Organize X: add a constant column for intercept
        x = np.column_stack([x_predicted, self.g,])
        x = sm.add_constant(x)  # add intercept

        # 2) Fit OLS
        model_sm = sm.OLS(self.y, x).fit()
        param = model_sm.params[1]  # pandas Series of parameter estimates
        conf_95 = model_sm.conf_int(0.05)[1]  # DataFrame with 2 columns [lower, upper]
        conf_90 = model_sm.conf_int(0.1)[1] # DataFrame with 2 columns [lower, upper]

        return param, conf_95, conf_90

    def estimate_sri(self):
        """
        Estimate the 2SLS model using a simple linear regression model - SRI
        :return: np.array, the coefficients of the Naive SR-IV model.
        """
        model = LinearRegression()

        model.fit(self.xa, self.x_1)
        x_predicted = model.predict(self.xb)
        x_error = self.x_2 - x_predicted

        # 1) Organize X: add a constant column for intercept
        x = np.column_stack([self.x_2, self.g, x_error])
        x = sm.add_constant(x)  # add intercept

        # 2) Fit OLS
        model_sm = sm.OLS(self.y, x).fit()
        param = model_sm.params[1]  # pandas Series of parameter estimates
        conf_95 = model_sm.conf_int(0.05)[1]  # DataFrame with 2 columns [lower, upper]
        conf_90 = model_sm.conf_int(0.1)[1] # DataFrame with 2 columns [lower, upper]

        return param, conf_95, conf_90

    def estimating_sri_sps_with_nn_with_confident_interval(self):
        """
        Estimate the 2SLS model using a simple neural network model. - SPS
        :return: tuple of np.array, the coefficients of the Naive SR-IV model.
        """
        model = DeepPLIV()
        # normalize the data
        scaler = StandardScaler()
        g_iv_1 = scaler.fit_transform(self.xa)
        g_iv_2 = scaler.transform(self.xb)
        x_1 = self.x_1
        x_2 = self.x_2

        first_stage_model = model.fit_first_stage(x_1, g_iv_1,
                                                  epochs_first_stage=self.epochs,
                                                  learning_rate_first_stage=self.learning_rate,
                                                  dropout=self.dropout,
                                                  validation_data = (g_iv_2, x_2))
        x_predicted = first_stage_model.predict(g_iv_2).reshape(1, -1)
        x_error = self.x_2 - x_predicted
        x_predicted = x_predicted
        scaler = StandardScaler()
        x_exog = scaler.fit_transform(np.concatenate((self.g.reshape(-1, 1),
                                                      x_error.reshape(-1, 1)), axis=1))
        # SRI model
        model_sri = model.fit_second_stage(self.x_2.reshape(-1, 1),
                                           x_exog,
                                           self.y.reshape(-1, 1),
                                           epochs_second_stage=self.epochs,
                                           learning_rate_second_stage=self.learning_rate,
                                           dropout=self.dropout)

        mode_sri_coef = model_sri.final_layer.weight.detach().numpy()[:, 0]


        # SPS model
        x_exog = self.g.reshape(-1, 1)

        model_sps = model.fit_second_stage(x_predicted.reshape(-1, 1),
                                          x_exog,
                                          self.y.reshape(-1, 1),
                                          epochs_second_stage=self.epochs,
                                          learning_rate_second_stage=self.learning_rate,
                                          dropout = self.dropout)

        model_sps_coef = model_sps.final_layer.weight.detach().numpy()[:, 0]
        # naive feed forward model
        model_nff = model.fit_second_stage(self.x_2.reshape(-1, 1),
                                             x_exog,
                                             self.y.reshape(-1, 1),
                                             epochs_second_stage=self.epochs,
                                             learning_rate_second_stage=self.learning_rate,
                                             dropout=self.dropout)
        model_nff_coef = model_nff.final_layer.weight.detach().numpy()[:, 0]

        return model_sps_coef, mode_sri_coef, model_nff_coef


def run_high_dimension_genetic_simulation(num_simulations=10,
                                          m: int = 1000,
                                          p_thr: float = 0.05,
                                          beta_1: float = 1,
                                          gamma_u: float = 1,
                                          beta_u: float = 1,
                                          beta_2: float = 1,
                                          n: int = 20000,
                                          learning_rate: float = 0.01,
                                          epochs: int = 2000,
                                          k: int = 5,
                                          dropout: float = 0.0, interaction=False):
    """
    Run the genetic simulation for the high-dimensional case.
     The simulation generates data using the SimDataCreatorHighDimension class
     and estimates the Naive SR-IV, SPS, and SRI models.
    :param num_simulations: parameter to control the number of simulations.
    :param beta_u: parameter to control the coefficient for the confounding variable.
    the endogenous and exogenous variables.
    :param beta_2: parameter to control the coefficient for the exogenous variable.
    :param beta_1: parameter to control the coefficient for the endogenous variable.
    :param n: parameter to control the number of samples.
    :param gamma_u: parameter to control the coefficient for the confounding variable.
    :param learning_rate: parameter to control the learning rate for the neural network model.
    :param epochs: parameter to control the number of epochs for training the neural network model.
    :param k: parameter to control the number of trainings for the neural network model.
    :param p_thr: parameter to control the threshold for the p-value.
    :param m: parameter to control the number of instrumental variables.
    :param dropout: parameter to control the dropout rate for the neural network model.
    :param interaction: parameter to control the interaction term between the endogenous and exogenous variables.
    :return: pd.DataFrame, the results of the simulation.
    """
    results = {
        'method': [],
        'coefficient': [],
        'value': []
    }

    for _ in range(num_simulations):
        betas = (beta_1, beta_2, beta_u)
        data_creator = SimDataCreatorHighDimension(n=n, m = m, betas=betas, gamma_u=gamma_u, p_thr=p_thr,
                                                   interaction=interaction)

        naive_srisps = NaiveSRISPSHighDimension(data_creator=data_creator, epochs=epochs, learning_rate=learning_rate,
                                                dropout=dropout)

        for method, estimate_func in [('Naive Regression', naive_srisps.estimate_naive_regression),
                                      ('SPS', naive_srisps.estimate_sps),
                                      ('SRI', naive_srisps.estimate_sri)]:
            coefficients = estimate_func()
            for i, coef in enumerate(coefficients):
                if i == 0:
                    results['method'].append(method)
                    results['coefficient'].append(f'coef_{i}')
                    results['value'].append(coef)
        coefficients_sri_lst, coefficients_sps_lst, coefficients_nff_lst = [], [], []
        for i in range(k):
            coefficients_sps, coefficients_sri, coefficients_nff = naive_srisps.estimating_sri_sps_with_nn_with_confident_interval()
            coefficients_sri_lst.append(coefficients_sri)
            coefficients_sps_lst.append(coefficients_sps)
            coefficients_nff_lst.append(coefficients_nff)
        coefficients_sri_mean = np.mean(coefficients_sri_lst, axis=0)
        coefficients_sps_mean = np.mean(coefficients_sps_lst, axis=0)
        coefficients_nff_mean = np.mean(coefficients_nff_lst, axis=0)

        # noinspection PyUnboundLocalVariable
        for method, coefficients in [('SPS with NN', coefficients_sps),
                                     ('SRI with NN', coefficients_sri),
                                     ('Naive Feed Forward', coefficients_nff)]:
            for i, coef in enumerate(coefficients):
                if i == 0:
                    if np.abs(coef)>10:
                        print(f'coef_{i} {method.lower()}: {coef}')
                        break
                    results['method'].append(method)
                    results['coefficient'].append(f'coef_{i}')
                    results['value'].append(coef)
                    print(f'coef_{i} {method.lower()}: {coef}')
        for method, coefficients in [(f'SPS with NN - mean {k}', coefficients_sps_mean),
                                     (f'SRI with NN - mean {k}', coefficients_sri_mean),
                                     (f'Naive Feed Forward - mean {k}', coefficients_nff_mean)]:
            for i, coef in enumerate(coefficients):
                if i == 0:
                    if np.abs(coef)>10:
                        print(f'coef_{i} {method.lower()}: {coef}')
                        break

    results_df = pd.DataFrame(results)
    # results_df.to_pickle(f'high_dim_sim_linear/{n}_{num_simulations}_{beta_1}_{beta_u}_{interaction}.pkl')
    return results_df



if __name__ == '__main__':
    _num_simulations: int = 1
    # This is the simulation for the first only the first part being non-linear
    _beta_1: float = 1
    _k: int = 1
    _lr: float = 0.005
    _gamma_u = 1
    _m = 200000
    _p_thr = 0.05e-6
    _beta_u = 2
    _interaction = True
    _n = 5000
    _dropout: float =  0.3
    _epochs: int = int((1.5 * 10 ** 7) / (_n//2))
    print(f'n: {_n}, dropout: {_dropout}, beta_u: {_beta_u},')

    res = run_high_dimension_genetic_simulation(num_simulations=_num_simulations,
                                                gamma_u=_gamma_u,
                                                beta_u=_beta_u,
                                                n=_n,
                                                m = _m,
                                                p_thr=_p_thr,
                                                beta_1=_beta_1,
                                                learning_rate=_lr,
                                                epochs = _epochs,
                                                k = _k,
                                                dropout=_dropout,
                                                interaction=_interaction)

