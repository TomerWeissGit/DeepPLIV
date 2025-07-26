import numpy as np

from models.first_stage import NeuralNetworkFirstStage
from models.second_stage import NeuralNetworkSecondStage

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

    def fit_first_stage(self,
                        z_1: np.array,
                        v_1: np.array,
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
        :param dropout: float, the dropout rate for the first stage.
        :param validation_data: tuple, the validation data.
        :param output_dim: int, the output dimension of the first stage model.
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