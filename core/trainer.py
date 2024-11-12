from models.first_stage import NeuralNetworkFirstStage
from models.second_stage import NeuralNetworkSecondStage
import numpy as np

class DeepPLIV:
    def __init__(self, z: np.array, v_1: np.array, v_2: np.array, y: np.array, x_1: np.array, x_2: np.array):
        """
        Initialize the DeepPLIV model.
        :param z_1: np.array, the instrumental variables training dataset.
        :param z_2: np.array, the instrumental variables for predicting v_2_hat.
        :param v_1: np.array, the endogenous variable/s training dataset.
        :param v_2: np.array, the endogenous variable/s for getting the error term v_2-v_2_hat.
        :param x_1: np.array, the first set of exogenous variables helping in predicting v_1.
        :param x_2: np.array, the second set of exogenous variables helping in predicting v_2_hat and y_hat.
        :param y: np.array, the dependent variable.

        """
        self.first_stage = NeuralNetworkFirstStage(input_dim_1)
        self.second_stage = NeuralNetworkSecondStage(input_dim_1, input_dim_2)