import numpy as np
from models.first_stage import NeuralNetworkFirstStage
from models.second_stage import NeuralNetworkSecondStage
from sklearn.linear_model import LinearRegression

import torch


class DeepPLIV:
    def __init__(self):
        self.first_stage_model: NeuralNetworkFirstStage
        self.second_stage_model: NeuralNetworkSecondStage
        pass

    def fit(self, v_1: np.array, z_1: np.array,z_2: np.array, x: np.array, y: np.array,
            first_stage_epochs: int = 500,
            first_stage_learning_rate: float = 0.01,
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
        :param second_stage_epochs: the number of epochs for training the second stage.
        :param second_stage_learning_rate: the learning rate for training the second stage.
        :return:
        """
        self.first_stage_model = self._fit_first_stage(v_1, z_1, first_stage_epochs, first_stage_learning_rate)
        v_hat = self._predict_first_stage(first_stage_model, z_2)
        self.second_stage_model = self._fit_second_stage(v_hat, x, y, second_stage_epochs, second_stage_learning_rate)
        return self.first_stage_model, self.second_stage_model

    def predict(self,
                z: np.array,
                x: np.array) -> np.array:
        """
        Predict the outcome variable using the DeepPLIV model.
        :param first_stage_model: the first stage model.
        :param second_stage_model: the second stage model.
        :param z: the instrumental variable for the second stage.
        :param x: the exogenous variable.
        :return: the predicted outcome variable.
        """
        v_hat = self._predict_first_stage(self.first_stage_model, z)
        return self._predict_second_stage(self.second_stage_model, v_hat, x)

    def _fit_first_stage(self, v_1: np.array, z_1: np.array,
                         epochs_first_stage: int,
                         learning_rate_first_stage: float) -> NeuralNetworkFirstStage:
        """
        Fit the DeepPLIV model.
        :param v_1: np.array, the dependent variable.
        :param z_1: np.array, the instrumental variable.
        :param epochs_first_stage: int, the number of epochs for training the first stage.
        :param learning_rate_first_stage: float, the learning rate for training the first stage.
        """

        self.first_stage_model = NeuralNetworkFirstStage(input_dim=z_1.shape[1])
        self.first_stage_model.train_new_data(z_1, v_1, epochs_first_stage, learning_rate_first_stage)
        return self.first_stage_model

    def _fit_second_stage(self, v_hat: np.array, x: np.array, y: np.array,
                          epochs_second_stage: int,
                          learning_rate_second_stage: float) -> NeuralNetworkSecondStage:
        """
        Fit the second stage of the DeepPLIV model.
        :param v_hat: np.array, the dependent variable prediction.
        :param x: np.array, the exogenous variable.
        :param epochs_second_stage: int, the number of epochs for training the second stage.
        :param learning_rate_second_stage: float, the learning rate for training the second stage.
        """
        self.second_stage_model = NeuralNetworkSecondStage(x=x.shape[1], v=v_hat.shape[1])
        self.second_stage_model.train_new_data(x_exog=x,
                                               v_linear=v_hat,
                                               y=y,
                                               epochs=epochs_second_stage,
                                               learning_rate=learning_rate_second_stage)
        return self.second_stage_model

    def _predict_first_stage(self, z_2: np.array) -> np.array:
        """
        Predict the endogenous variable.
        :param z_2: np.array, the instrumental variable.
        :return: np.array, the predicted endogenous variable.
        """
        return self.first_stage_model.predict(z_2)

    def _predict_second_stage(self, v_hat: np.array, x: np.array) -> np.array:
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


if __name__ == '__main__':
    from examples.example_usage import SimDataCreator

    deep_pliv = DeepPLIV()
    p = 100
    n = 500
    beta_1 = 0.5
    beta_u = 10
    gamma_u = 2
    std_epsilon = 3
    std_eta = 3
    scenario = 5
    data = SimDataCreator(p, n, beta_1, beta_u, gamma_u, std_epsilon, std_eta, scenario)
    v = data.v
    z = data.z
    first_stage_epochs = 5000
    first_stage_learning_rate = 0.05
    first_stage_model = deep_pliv._fit_first_stage(v, z, first_stage_epochs, first_stage_learning_rate)
    first_stage_2sls_model = LinearRegression()
    first_stage_2sls_model.fit(z, v)
    data_2 = SimDataCreator(p, n, beta_1, beta_u, gamma_u, std_epsilon, std_eta, scenario)
    z_second = data_2.z
    z_predicted_linear_regression = first_stage_2sls_model.predict(z_second)
    v_predicted = deep_pliv._predict_first_stage(z_second)

    v_pred_error_linear_regression = z_predicted_linear_regression - data_2.v
    v_pred_error = v_predicted-data_2.v
    predicted_error = np.sqrt((v_pred_error ** 2).mean())
    print(f"Predicted error: {predicted_error}")
    x = np.concatenate((v_pred_error, np.ones((n, 1))), axis=1)
    y = data_2.y
    v = data_2.v.reshape(-1, 1)
    second_stage_epochs = 5000
    second_stage_learning_rate = 0.01
    # v_predicted = np.concatenate((v_predicted, np.ones((n, 1))), axis=1)

    second_stage_linear_model = LinearRegression()
    x_linear_regression = np.concatenate(((v_pred_error_linear_regression - data_2.v).reshape(-1, 1), np.ones((n, 1))),
                                         axis=1)
    x_all = np.concatenate((x_linear_regression, v), axis=1)
    second_stage_linear_model.fit(x_all, y)

    second_stage_model = deep_pliv._fit_second_stage(v, x, y, second_stage_epochs, second_stage_learning_rate)
    y_predicted = deep_pliv._predict_second_stage(v, x)
    predicted_error = np.sqrt(((y_predicted - y) ** 2).mean())
    print('deep_coefs:', deep_pliv.get_v_predicted_coefficient())
    print('linear_coefs:', second_stage_linear_model.coef_[-1])
    print(f"Predicted error deep: {predicted_error}")
    print(f"Predicted error linear: {np.sqrt(((second_stage_linear_model.predict(x_all) - y) ** 2).mean())}")


