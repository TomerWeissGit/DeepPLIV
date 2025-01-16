import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from core.trainer import DeepPLIV
from sklearn.preprocessing import StandardScaler

class SimDataCreatorHighDimension:
    def __init__(self,
                 n: int,
                 beta_1: float,
                 rho: float = 0.1):
        """
        Initialize the SimDataCreatorHighDimension class.

        :param n: int, number of samples.
        :param beta_1: float, coefficient for the endogenous variable.
        """
        self.n_x_1 = self.n_x_2 = self.n_y = n // 2
        self.v = np.random.normal(0, 1, n)
        self.e = np.array([np.random.normal(-5 * rho * v, 1 - rho ** 2, 1) for v in self.v[self.n_x_1:]])[:, 0]
        self.beta_1 = beta_1
        self.x_1, self.x_2, self.t_1, self.t_2, self.z_1, self.z_2 = self._generate_x_df()
        self.y, self.s = self._generate_y_df_linear()

    def _generate_x_df(self):
        error_1 = self.v[:self.n_x_1]
        error_2 = self.v[self.n_x_1:]
        t_1 = np.array([random.randint(1, 10) for _ in range(self.n_x_1)])
        t_2 = np.array([random.randint(1, 10) for _ in range(self.n_x_2)])
        z_1 = np.random.normal(0, 1, self.n_x_1)
        z_2 = np.random.normal(0, 1, self.n_x_2)
        scaler = StandardScaler()
        x_1 = scaler.fit_transform((25 + np.array([self._ft(t) for t in t_1]) * (z_1 + 3) ).reshape(-1, 1))[:, 0] + error_1
        x_2 = scaler.transform((25 + np.array([self._ft(t) for t in t_2]) * (z_2 + 3)).reshape(-1, 1))[:, 0] + error_2

        return x_1, x_2 , t_1, t_2 , z_1 , z_2

    def _generate_y_df_linear(self):
        s = np.array([random.randint(1, 7) for _ in range(self.n_y)])
        scaler = StandardScaler()
        y = scaler.fit_transform((100 + (10 + self.x_2) * s * self._ft(self.t_2)).reshape(-1, 1))[:, 0]
        y = y + beta_1 * self.x_2 + self.e# scale the data
        return y, s


    def get_data(self):
        return self.x_1, self.x_2, self.y, self.s, self.z_1, self.z_2, self.t_1, self.t_2

    @staticmethod
    def _ft(t):
        return 2.0 * ((t - 5) ** 4 / 600 + np.exp(-((t - 5) / 0.5) ** 2) + t / 10. - 2)


class NaiveSRISPSHighDimension:
    def __init__(self,
                 data_creator: SimDataCreatorHighDimension,
                 epochs: int = 1000,
                 learning_rate: float = 0.001,
                 dropout: float = 0):
        """
        Initialize the NaiveSRISPSHighDimension class.
        :param data_creator: SimDataCreatorHighDimension, an instance of the SimDataCreatorHighDimension class.
        :param epochs: int, the number of epochs for training the neural network model.
        :param learning_rate: float, the learning rate for training the neural network model.
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

        first_stage_model = model.fit_first_stage_mdn(x_1, g_iv_1,
                                                  epochs_first_stage=self.epochs,
                                                  learning_rate_first_stage=self.learning_rate,
                                                  dropout=self.dropout,
                                                  validation_data = (g_iv_2, x_2))
        x_predicted = first_stage_model.predict_mean(g_iv_2)
        x_error = self.x_2 - x_predicted
        x_predicted = x_predicted
        scaler = StandardScaler()
        x_exog = np.concatenate((self.s.reshape(-1, 1),
                                                      self.t_2.reshape(-1, 1),
                                                      x_error.reshape(-1, 1)), axis=1)
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


def run_high_dimension_genetic_simulation(num_simulations=10,
                                          beta_1: float = 1,
                                          n: int = 20000,
                                          learning_rate: float = 0.01,
                                          epochs: int = 2000,
                                          k: int = 5,
                                          dropout: float = 0.0,
                                          rho: float = 0.1):
    """
    Run the genetic simulation for the high-dimensional case.
     The simulation generates data using the SimDataCreatorHighDimension class
     and estimates the Naive SR-IV, SPS, and SRI models.
    :param num_simulations: parameter to control the number of simulations.
    :param beta_u: parameter to control the coefficient for the confounding variable.
    :param beta_3: parameter to control the coefficient for the interaction term between
    the endogenous and exogenous variables.
    :param beta_2: parameter to control the coefficient for the exogenous variable.
    :param beta_1: parameter to control the coefficient for the endogenous variable.
    :param n: parameter to control the number of samples.
    :param gamma_u: parameter to control the coefficient for the confounding variable.
    :param learning_rate: parameter to control the learning rate for the neural network model.
    :param epochs: parameter to control the number of epochs for training the neural network model.
    :param k: parameter to control the number of trainings for the neural network model.
    :return: pd.DataFrame, the results of the simulation.
    """
    results = {
        'method': [],
        'coefficient': [],
        'value': []
    }

    for _ in range(num_simulations):
        data_creator = SimDataCreatorHighDimension(n=n, beta_1=beta_1, rho=rho)

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
                        results['method'].append(method)
                        results['coefficient'].append(f'coef_{i}')
                        results['value'].append(coef)
                        print(f'coef_{i} {method.lower()}: {coef}')

    results_df = pd.DataFrame(results)
    results_df.to_pickle(f'deep_iv_sim/{n}_{num_simulations}_{beta_1}_{rho}.pkl')
    return results_df



if __name__ == '__main__':
    num_simulations: int = 50
    # This is the simulation for the first only the first part being non-linear
    beta_1: float = -2
    k: int = 1
    lr: float = 0.001
    for n in [2000, 10000, 20000, 40000]:
        for rho in [0.5]:
            dropout: float = 1000 / (1000 + n//2)
            epochs: int = int((1.5 * 10 ** 7) / (n//2))
            print(f'n: {n}, dropout: {dropout}, rho: {rho},')
            res = run_high_dimension_genetic_simulation(num_simulations=num_simulations,
                                                        n=n,
                                                        rho=rho,
                                                        beta_1=beta_1,
                                                        learning_rate=lr,
                                                        epochs = epochs,
                                                        k = k,
                                                        dropout=dropout)
            # res = pd.read_pickle(f'deep_iv_sim/{n}_{num_simulations}_{beta_1}_{rho}.pkl')
            # plot_boxplot(res, y_line=beta_1)





