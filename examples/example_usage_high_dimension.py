import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from core.trainer import DeepPLIV
from utils.helpers import plot_boxplot
from typing import Literal
from sklearn.preprocessing import StandardScaler

class SimDataCreatorHighDimension:
    def __init__(self, n, p, beta_1, beta_2, beta_3, beta_u, gamma_u, non_null_iv, gwas_threshold=None, scenario=1):
        self.n_x_1 = self.n_x_2 = self.n_y = n // 2
        self.u = np.random.normal(0, 1, n)
        self.p = p
        self.beta_1, self.beta_2, self.beta_3, self.beta_u, self.gamma_u = beta_1, beta_2, beta_3, beta_u, gamma_u
        self.non_null_iv = non_null_iv
        self.gamma_iv = np.zeros(p)
        self.g_iv = np.random.binomial(2, 0.3, size=(n, p)) / 2
        self.g_iv_1, self.g_iv_2 = self.g_iv[:self.n_x_1], self.g_iv[self.n_x_1:]
        self.g = np.random.binomial(2, 0.3, size=self.n_y)
        self.gwas_threshold = 5 / np.sqrt(self.n_x_1) if gwas_threshold is None else gwas_threshold
        self._generate_gamma_iv()
        self.x_1, self.x_2 = self._generate_x_df(scenario)
        self.y = self._generate_y_df()

    def _generate_gamma_iv(self):
        random_gammas_gt = []
        random_gammas = []
        while len(random_gammas_gt) < self.non_null_iv:
            random_gammas = np.random.normal(0, self.gwas_threshold / 2, 1000000)
            random_gammas_gt += list(random_gammas[np.abs(random_gammas) > self.gwas_threshold])
        self.gamma_iv[:self.non_null_iv] = np.random.choice(random_gammas_gt, self.non_null_iv, replace=False)
        self.gamma_iv[self.non_null_iv:] = np.random.choice(random_gammas[np.abs(random_gammas) < self.gwas_threshold],
                                                            len(self.gamma_iv[self.non_null_iv:]))

    @staticmethod
    def _standardize(x_1, x_2):
        scaler = StandardScaler()
        x_1 = scaler.fit_transform(x_1.reshape(-1, 1)).flatten()
        x_2 = scaler.transform(x_2.reshape(-1, 1)).flatten()
        return x_1, x_2

    def _generate_x_df(self, scenario):
        error_1, error_2 = np.random.normal(0, 1, self.n_x_1), np.random.normal(0, 1, self.n_x_2)
        if scenario == 1:
            x_1 = self.g_iv_1 @ self.gamma_iv + self.gamma_u * self.u[:self.n_x_1] + error_1
            x_2 = self.g_iv_2 @ self.gamma_iv + self.gamma_u * self.u[self.n_x_1:] + error_2
        elif scenario == 2:
            x_1 = (np.abs(self.g_iv_1 - 0.75) * 2) @ self.gamma_iv + self.gamma_u * self.u[:self.n_x_1] + error_1
            x_2 = (np.abs(self.g_iv_2 - 0.75) * 2) @ self.gamma_iv + self.gamma_u * self.u[self.n_x_1:] + error_2
        elif scenario == 3:
            x_1 = (self.g_iv_1 ** 2) @ self.gamma_iv + self.gamma_u * self.u[:self.n_x_1] + error_1
            x_2 = (self.g_iv_2 ** 2) @ self.gamma_iv + self.gamma_u * self.u[self.n_x_1:] + error_2
        elif scenario == 4:
            x_1 = (self.g_iv_1==0.5).astype(float) @ self.gamma_iv + self.gamma_u * self.u[:self.n_x_1] + error_1
            x_2 = (self.g_iv_2==0.5).astype(float) @ self.gamma_iv + self.gamma_u * self.u[self.n_x_1:] + error_2
        elif scenario == 5:
            x_1 = (np.exp(self.g_iv_1)).astype(float) @ self.gamma_iv + self.gamma_u * self.u[:self.n_x_1] + error_1
            x_2 = (np.exp(self.g_iv_2)).astype(float) @ self.gamma_iv + self.gamma_u * self.u[self.n_x_1:] + error_2
        elif scenario == 6:
            x_1 = self.g_iv_1[:, 0] * self.g_iv_1[:, 1] + self.g_iv_1[:, 2] * self.g_iv_1[:, 3] + self.g_iv_1 @ self.gamma_iv + self.gamma_u * self.u[:self.n_x_1] + error_1
            x_2 = self.g_iv_2[:, 0] * self.g_iv_2[:, 1] + self.g_iv_1[:, 2] * self.g_iv_1[:, 3] + self.g_iv_2 @ self.gamma_iv + self.gamma_u * self.u[self.n_x_1:] + error_2

            # Initialize x_1 and x_2 as zeros

            return x_1, x_2
        else:
            raise ValueError('Scenario not implemented')
        return self._standardize(x_1, x_2)

    def _generate_y_df(self):
        y_base = self.beta_2 * self.g + self.beta_u * self.u[self.n_x_1:] + np.random.normal(0, 1, self.n_y)
        return self.beta_1 * self.x_2 + y_base + self.beta_3 * (self.g * self.x_2)

    def get_data(self):
        return self.x_1, self.x_2, self.y, self.g_iv_1, self.g_iv_2, self.g


class NaiveSRISPSHighDimension:
    def __init__(self, data_creator: SimDataCreatorHighDimension, epochs: int = 1000, learning_rate: float = 0.01):
        """
        Initialize the NaiveSRISPSHighDimension class.
        :param data_creator: SimDataCreatorHighDimension, an instance of the SimDataCreatorHighDimension class.
        :param epochs: int, the number of epochs for training the neural network model.
        :param learning_rate: float, the learning rate for training the neural network model.
        """
        self.data_creator = data_creator
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.x_1, self.x_2, self.y, self.g_iv_1, self.g_iv_2, self.g = data_creator.get_data()

    def estimate_naive_regression(self):
        """
        Estimate the Naive SR-IV model using a simple linear regression model.
        :return: np.array, the coefficients of the Naive SR-IV model.
        """
        x = np.concatenate((self.x_2.reshape(-1,1), self.g.reshape(-1, 1), (self.x_2 * self.g).reshape(-1,1)), axis=1)
        model = LinearRegression()
        model.fit(x, self.y)
        return model.coef_

    def estimate_sps(self):
        """
        Estimate the 2SLS model using a simple linear regression model. - SPS
        :return: np.array, the coefficients of the Naive SR-IV model.
        """

        model = LinearRegression()
        model.fit(self.g_iv_1, self.x_1)
        x_predicted = model.predict(self.g_iv_2)
        x = np.concatenate((x_predicted.reshape(-1, 1),
                            self.g.reshape(-1, 1),
                            (x_predicted * self.g).reshape(-1, 1)),
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
        model.fit(self.g_iv_1, self.x_1)
        x_predicted = model.predict(self.g_iv_2)
        x_error = self.x_2 - x_predicted

        x = np.concatenate([self.x_2.reshape(-1, 1),
                           self.g.reshape(-1, 1),
                           (self.x_2 * self.g).reshape(-1, 1),
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
        g_iv_1 = scaler.fit_transform(self.g_iv_1)
        g_iv_2 = scaler.transform(self.g_iv_2)
        x_1 = self.x_1.copy()

        first_stage_model = model.fit_first_stage(x_1, g_iv_1,
                                                  epochs_first_stage=self.epochs,
                                                  learning_rate_first_stage=self.learning_rate,
                                                  validation_data = (g_iv_2, self.x_2))
        x_predicted = first_stage_model.predict(g_iv_2)
        x_error = self.x_2.reshape(-1, 1) - x_predicted

        # SRI model
        x_sri = np.concatenate((self.x_2.reshape(-1, 1), self.g.reshape(-1, 1), (self.x_2 * self.g).reshape(-1, 1), x_error), axis=1)
        model_sri = LinearRegression()
        model_sri.fit(x_sri, self.y)

        # SPS model
        x_sps = np.concatenate((x_predicted, self.g.reshape(-1, 1), x_predicted * self.g.reshape(-1, 1)), axis=1)
        model_sps = LinearRegression()
        model_sps.fit(x_sps, self.y)

        return model_sri.coef_, model_sps.coef_


def run_high_dimension_genetic_simulation(num_simulations=10,
                                          beta_u: float = 0,
                                          beta_3: float = 0,
                                          beta_2: float = 0.5,
                                          beta_1: float = 1,
                                          scenario: Literal[1, 2, 3] =1,
                                          n: int = 5000,
                                          p: int = 1000,
                                          gamma_u: float = 1.0,
                                          non_null_iv: int = 20,
                                          learning_rate: float = 0.01,
                                          epochs: int = 2000,
                                          gwas_threshold: float = None,
                                          k: int = 5):
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
    :param scenario: parameter to control the scenario to consider.
    :param n: parameter to control the number of samples.
    :param p: parameter to control the number of features (in the first stage).
    :param gamma_u: parameter to control the coefficient for the confounding variable.
    :param non_null_iv: parameter to control the number of non-null instrumental variables.
    :param learning_rate: parameter to control the learning rate for the neural network model.
    :param epochs: parameter to control the number of epochs for training the neural network model.
    :param gwas_threshold: parameter to control the threshold for the GWAS.
    :return: pd.DataFrame, the results of the simulation.
    """
    results = {
        'method': [],
        'coefficient': [],
        'value': []
    }

    for _ in range(num_simulations):
        data_creator = SimDataCreatorHighDimension(n=n, p=p, beta_1=beta_1, beta_2=beta_2, beta_3=beta_3, beta_u=beta_u,
                                                   gamma_u=gamma_u, non_null_iv=non_null_iv, scenario=scenario,
                                                   gwas_threshold=gwas_threshold)

        naive_srisps = NaiveSRISPSHighDimension(data_creator=data_creator, epochs=epochs, learning_rate=learning_rate)

        for method, estimate_func in [('Naive Regression', naive_srisps.estimate_naive_regression),
                                      ('SPS', naive_srisps.estimate_sps),
                                      ('SRI', naive_srisps.estimate_sri)]:
            coefficients = estimate_func()
            for i, coef in enumerate(coefficients):
                results['method'].append(method)
                results['coefficient'].append(f'coef_{i}')
                results['value'].append(coef)
        coefficients_sri_lst, coefficients_sps_lst = [], []
        for i in range(k):
            coefficients_sri, coefficients_sps = naive_srisps.estimating_sri_sps_with_nn()
            coefficients_sri_lst.append(coefficients_sri)
            coefficients_sps_lst.append(coefficients_sps)
        coefficients_sri_mean = np.mean(coefficients_sri_lst, axis=0)
        coefficients_sps_mean = np.mean(coefficients_sps_lst, axis=0)
        for method, coefficients in [('SRI with NN', coefficients_sri), ('SPS with NN', coefficients_sps)]:
            for i, coef in enumerate(coefficients):
                if np.abs(coef)>10:
                    print(f'coef_{i} {method.lower()}: {coef}')
                    break
                results['method'].append(method)
                results['coefficient'].append(f'coef_{i}')
                results['value'].append(coef)
                if i == 0:
                    print(f'coef_{i} {method.lower()}: {coef}')
        for method, coefficients in [(f'SRI with NN - mean {k}', coefficients_sri_mean),
                                     (f'SPS with NN - mean {k}', coefficients_sps_mean)]:
            for i, coef in enumerate(coefficients):
                if np.abs(coef)>10:
                    print(f'coef_{i} {method.lower()}: {coef}')
                    break
                results['method'].append(method)
                results['coefficient'].append(f'coef_{i}')
                results['value'].append(coef)
                if i == 0:
                    print(f'coef_{i} {method.lower()}: {coef}')

    results_df = pd.DataFrame(results)
    results_df.to_pickle(f'high_dim_sim_results_high_dropout_{num_simulations}_{beta_u}_{beta_3}_{scenario}_{non_null_iv}_with_{k}_trainings.pkl')
    return results_df

if __name__ == '__main__':
    #  This is the simulation for the first only the first part being non-linear
    # num_simulations: int = 300
    # beta_2: float = 1
    # beta_1: float = 0.5
    # n: int =  20000
    # p: int = 500
    # k: int = 10
    # # lr: float = n * 5e-8 / p
    # lr: float = 0.001
    # gamma_u : float = 1
    # epochs : int = 100
    # gwas_threshold: float = 0.05
    # for scenario in [2]:
    #     for non_null_iv in [10, 100, 200, 500]:
    #         print(scenario, non_null_iv)
    #         for beta_u_ in [1]:
    #             for beta_3_ in [0.5]:
    #                 res = run_high_dimension_genetic_simulation(num_simulations=num_simulations,
    #                                                             beta_u=beta_u_,
    #                                                             beta_3=beta_3_,
    #                                                             scenario=scenario,
    #                                                             p=p,
    #                                                             n=n,
    #                                                             beta_1=beta_1,
    #                                                             beta_2=beta_2,
    #                                                             gamma_u=gamma_u,
    #                                                             non_null_iv = non_null_iv,
    #                                                             learning_rate=lr,
    #                                                             epochs = epochs,
    #                                                             gwas_threshold=gwas_threshold,
    #                                                             k = k)
    #
    #                 # plot_boxplot(res, y_line=beta_1)

