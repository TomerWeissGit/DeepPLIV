from os.path import exists

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from core.trainer import DeepPLIV
from sklearn.preprocessing import StandardScaler
from data_creation import SimDataCreatorDeepIV
from utils.helpers import plot_boxplot

class NaiveSRISPSHighDimension:
    def __init__(self,
                 data_creator: SimDataCreatorDeepIV,
                 epochs: int = 1000,
                 learning_rate: float = 0.001,
                 dropout: float = 0):
        """
        Initialize the NaiveSRISPSHighDimension class.
        :param data_creator: SimDataCreatorDeepIV, an instance of the SimDataCreatorDeepIV class.
        :param epochs: int, the number of epochs for training the neural network model.
        :param learning_rate: float, the learning rate for training the neural network model.
        :param dropout: float, the dropout rate for the neural network model.
        """
        self.data_creator = data_creator
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.x_1, self.x_2, self.y, self.s, self.z_1, self.z_2, self.t_1, self.t_2 = data_creator.get_data()
        self.dropout = dropout
        self.xa = np.concatenate((self.z_1.reshape(-1, 1), self.t_1.reshape(-1, 1)), axis=1)
        self.xb = np.concatenate((self.z_2.reshape(-1, 1), self.t_2.reshape(-1, 1)), axis=1)

    def estimate_naive_regression(self):
        """
        Estimate the Naive SR-IV model using a simple linear regression model.
        :return: np.array, the coefficients of the Naive SR-IV model.
        """
        x = np.concatenate((self.x_2.reshape(-1, 1),
                            self.t_2.reshape(-1, 1),
                            self.s.reshape(-1, 1),
                            (self.t_2 * self.x_2 * self.s).reshape(-1, 1)), axis=1)
        model = LinearRegression()
        model.fit(x, self.y)
        return model.coef_

    def estimate_sps(self):
        """
        Estimate the 2SLS model using a simple linear regression model. - SPS
        :return: np.array, the coefficients of the Naive SR-IV model.
        """
        model = LinearRegression()

        model.fit(self.xa, self.x_1)
        x_predicted = model.predict(self.xb)
        x = np.concatenate((x_predicted.reshape(-1, 1),
                            self.t_2.reshape(-1, 1),
                            self.s.reshape(-1, 1),
                            (x_predicted * self.t_2 * self.s).reshape(-1, 1)),
                           axis=1)
        model = LinearRegression()
        model.fit(x, self.y)
        return model.coef_

    def estimate_sri(self):
        """
        Estimate the 2SLS model using a simple linear regression model - SRI
        :return: np.array, the coefficients of the Naive SR-IV model.
        """
        model = LinearRegression()

        model.fit(self.xa, self.x_1)
        x_predicted = model.predict(self.xb)
        x_error = self.x_2 - x_predicted

        x = np.concatenate([self.x_2.reshape(-1, 1),
                           self.t_2.reshape(-1, 1),
                            self.s.reshape(-1, 1),
                           (self.x_2 * self.t_2 * self.s).reshape(-1, 1),
                x_error.reshape(-1, 1)],
                           axis =1 )
        model = LinearRegression()
        model.fit(x, self.y)
        return model.coef_

    def estimating_sri_sps_with_nn(self):
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
        x_exog = scaler.fit_transform(np.concatenate((self.s.reshape(-1, 1),
                                                      self.t_2.reshape(-1, 1),
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
        scaler = StandardScaler()
        x_exog = scaler.fit_transform(np.concatenate((self.s.reshape(-1, 1),
                                                      self.t_2.reshape(-1, 1)), axis=1))

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


def run_deep_iv_simulation(num_simulations=10,
                                          beta_1: float = 1,
                                          n: int = 20000,
                                          learning_rate: float = 0.01,
                                          epochs: int = 2000,
                                          k: int = 5,
                                          dropout: float = 0.0,
                                          rho: float = 0.1):
    """
    Run the genetic simulation for the high-dimensional case.
     The simulation generates data using the SimDataCreatorDeepIV class
     and estimates the Naive SR-IV, SPS, and SRI models.
    :param num_simulations: parameter to control the number of simulations.
    the endogenous and exogenous variables.
    :param beta_1: parameter to control the coefficient for the endogenous variable.
    :param n: parameter to control the number of samples.
    :param learning_rate: parameter to control the learning rate for the neural network model.
    :param epochs: parameter to control the number of epochs for training the neural network model.
    :param k: parameter to control the number of trainings for the neural network model.
    :param dropout: parameter to control the dropout rate for the neural network model.
    :param rho: parameter to control the correlation between the endogenous and instrumental variables.
    :return: pd.DataFrame, the results of the simulation.
    """
    if exists(f'deep_iv_sim_smaller_model/{n}_{num_simulations}_{beta_1}_{rho}.pkl'):
        return pd.read_pickle(f'deep_iv_sim_smaller_model/{n}_{num_simulations}_{beta_1}_{rho}.pkl')
    results = {
        'method': [],
        'coefficient': [],
        'value': []
    }

    for _ in range(num_simulations):
        data_creator = SimDataCreatorDeepIV(n=n, beta_1=beta_1, rho=rho)

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
            coefficients_sps, coefficients_sri, coefficients_nff = naive_srisps.estimating_sri_sps_with_nn()
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
    results_df.to_pickle(f'deep_iv_sim_smaller_model/{n}_{num_simulations}_{beta_1}_{rho}.pkl')
    return results_df


def run_single_config(n, rho):
    _num_simulations = 300
    _beta_1 = -2
    _k = 1
    _lr = 0.01

    _dropout = 100 / (1000 + n // 2)
    _epochs = int((1.5 * 10 ** 7) / (n // 2))

    print(f"n: {n}, dropout: {_dropout}, rho: {rho}")

    return run_deep_iv_simulation(
        num_simulations=_num_simulations,
        n=n,
        rho=rho,
        beta_1=_beta_1,
        learning_rate=_lr,
        epochs=_epochs,
        k=_k,
        dropout=_dropout
    )


if __name__ == '__main__':
    import concurrent.futures

    _n_values = [2000, 10000, 20000, 40000]
    _rho_values = [0, 0.1, 0.25, 0.5, 0.75, 0.9]

    configs = [(n, rho) for n in _n_values for rho in _rho_values]

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_config, n, rho) for n, rho in configs]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            # res = pd.read_pickle(f'deep_iv_sim_smaller_model/{_n}_{_num_simulations}_{_beta_1}_{_rho}.pkl')
            # plot_boxplot(res, y_line=_beta_1)
