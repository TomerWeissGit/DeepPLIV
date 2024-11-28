import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from core.trainer import DeepPLIV
from utils.helpers import plot_boxplot
from typing import Literal

class SimDataCreatorHighDimension:
    def __init__(self,
                 n: int,
                 p: int,
                 beta_1: float,
                 beta_2: float,
                 beta_3: float,
                 beta_u: float,
                 gamma_u: float,
                 non_null_iv: int,
                 gwas_threshold: float = 5e-8,
                 scenario: int = 1):
        """
        Initialize the SimDataCreatorHighDimension class.

        :param n: int, number of samples.
        :param p: int, number of features.
        :param beta_1: float, coefficient for the endogenous variable.
        :param beta_2: float, coefficient for the exogenous variable.
        :param beta_3: float, coefficient for the interaction term between the endogenous and exogenous variables.
        :param beta_u: float, coefficient for the confounding variable.
        :param gamma_u: float, coefficient for the confounding variable.
        :param non_null_iv: int, indices of the non-null instrumental variables.
        """
        self.n_x_1 = n//2
        self.n_x_2 = n//2
        self.n_y = n//2
        self.std_x = np.sqrt(1/self.n_x_1)
        # Generate the confounding variable
        self.u = np.random.normal(0, 1, n)
        self.p = p
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.beta_u = beta_u
        self.gamma_u = gamma_u
        self.non_null_iv = non_null_iv

        # create the instrumental variables
        self.gamma_iv = np.zeros(p)
        self.g_iv = np.random.binomial(2, 0.3, size=(n, p))
        # split for first and second batch
        self.g_iv_1 = self.g_iv[:self.n_x_1]
        self.g_iv_2 = self.g_iv[self.n_x_1:]
        # create the exogenous variable
        self.g = np.random.binomial(2, 0.3, size=self.n_y)
        self.g = self.g - self.g.mean()

        self.gwas_threshold = gwas_threshold
        self._generate_gamma_iv()
        self.x_1, self.x_2 = self._generate_x_df(scenario)
        self.y = self._generate_y_df()

    def _generate_gamma_iv(self):
        random_gammas = np.random.normal(0, self.std_x, 10000000)
        random_strong_ivs = [np.random.randint(0,
                                               len(random_gammas[np.abs(random_gammas) > self.gwas_threshold]),
                                               self.non_null_iv)]
        random_weak_ivs = [np.random.randint(0,
                                             len(random_gammas[np.abs(random_gammas) < self.gwas_threshold]),
                                             len(self.gamma_iv[self.non_null_iv:]))]
        self.gamma_iv[:self.non_null_iv] = random_gammas[np.abs(random_gammas)>self.gwas_threshold][random_strong_ivs]
        self.gamma_iv[self.non_null_iv:] = random_gammas[np.abs(random_gammas)<self.gwas_threshold][random_weak_ivs]

    @staticmethod
    def _standardize(x_1, x_2):
        return x_1 / x_1.std(), x_2 / x_1.std() # dividing by the same std is important

    def _generate_x_df(self, scenario: Literal[1, 2, 3] = 1):
        """
        Generate the data for the exogenous variables.
        :param scenario: int, the scenario to consider.
        :return: pd.DataFrame, the data for the exogenous variables.
        """
        # scenario_1 - linear
        if scenario == 1:
            x_1 = self.g_iv_1 @ self.gamma_iv + self.gamma_u * self.u[:self.n_x_1] + np.random.normal(0, 1, self.n_x_1)
            x_2 = self.g_iv_2 @ self.gamma_iv + self.gamma_u * self.u[self.n_x_1:] + np.random.normal(0, 1, self.n_x_2)
            return self._standardize(x_1, x_2)
        # scenario_2 - non-linear 1 - same effect for 1 or 2 snps.
        if scenario == 2:
            def _aux_non_linear_1(g_iv, gamma_iv, gamma_u, u, n_x):
                return (np.abs(g_iv-1.5)*2) @ gamma_iv + gamma_u * u + np.random.normal(0, 1, n_x)
            x_1 = _aux_non_linear_1(self.g_iv_1, self.gamma_iv, self.gamma_u,
                                                 self.u[:self.n_x_1], self.n_x_1)
            x_2 = _aux_non_linear_1(self.g_iv_2, self.gamma_iv, self.gamma_u, self.u[self.n_x_1:],
                                                 self.n_x_2)
            return self._standardize(x_1, x_2)
        # scenario_3 - non-linear 2 - having 2 snps dramatically increase effect, not in a linear way
        if scenario == 3:
            def _aux_non_linear_2(g_iv, gamma_iv, gamma_u, u, n_x):
                return (g_iv**2) @ gamma_iv + gamma_u * u + np.random.normal(0, 1, n_x)
            x_1 = _aux_non_linear_2(self.g_iv_1, self.gamma_iv, self.gamma_u,
                                                 self.u[:self.n_x_1], self.n_x_1)
            x_2 = _aux_non_linear_2(self.g_iv_2, self.gamma_iv, self.gamma_u, self.u[self.n_x_1:],
                                                 self.n_x_2)
            return self._standardize(x_1, x_2)
        else:
            raise ValueError('Scenario not implemented')


    def _generate_y_df(self):
        y_base = self.beta_2 * self.g + self.beta_u * self.u[self.n_x_1:] + np.random.normal(0, 1, self.n_y)

        def aux_generate_y(x):
            return self.beta_1 * x + y_base + self.beta_3 * (self.g * x)

        y = aux_generate_y(self.x_2)
        return y

    def get_data(self):
        """
        Get the data generated by the class. The data is in the form of a pandas DataFrame. The columns are as follows:
        - x_df: pd.DataFrame, the exogenous variables.
        - y: pd.DataFrame, the endogenous variable.
        - g_iv: the instrumental variables.
        - g: exogenous variable.
        :return: tuple of pd.DataFrame, the generated data.(x_df, y, g_iv, g)
        """
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
        x = np.concatenate((self.x_2, self.g.reshape(-1, 1), self.x_2 * self.g.reshape(-1, 1)), axis=1)
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
        x = np.concatenate((x_predicted, self.g.reshape(-1, 1), x_predicted * self.g.reshape(-1, 1)), axis=1)
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


        x = np.concatenate((self.x_2, self.g.reshape(-1, 1), self.x_2 * self.g.reshape(-1, 1), x_error), axis=1)
        model = LinearRegression()
        model.fit(x, self.y)
        return model.coef_

    def estimating_sri_sps_with_nn(self):
        """
        Estimate the 2SLS model using a simple neural network model. - SPS
        :return: tuple of np.array, the coefficients of the Naive SR-IV model.
        """
        model = DeepPLIV()
        vars_first_stage_normalized = (self.g_iv_1 - self.g_iv_1.mean(axis=0)) / self.g_iv_1.std(axis=0)
        vars_second_stage_normalized = (self.g_iv_2 - self.g_iv_2.mean(axis=0)) / self.g_iv_1.std(axis=0)
        first_stage_model = model.fit_first_stage(self.x_1, vars_first_stage_normalized,
                                                  epochs_first_stage=self.epochs,
                                                  learning_rate_first_stage=self.learning_rate,
                                                  validation_data = (vars_second_stage_normalized, self.x_2))
        x_predicted = first_stage_model.predict(vars_second_stage_normalized)
        x_error = self.x_2 - x_predicted

        # SRI model
        x_sri = np.concatenate((self.x_2, self.g.reshape(-1, 1), self.x_2 * self.g.reshape(-1, 1), x_error), axis=1)
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
                                          epochs: int = 2000):
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
    :return: pd.DataFrame, the results of the simulation.
    """
    results = {
        'method': [],
        'coefficient': [],
        'value': []
    }


    for _ in range(num_simulations):
        data_creator = SimDataCreatorHighDimension(n=n, p=p, beta_1=beta_1, beta_2=beta_2, beta_3=beta_3, beta_u=beta_u,
                                                   gamma_u=gamma_u, non_null_iv=non_null_iv, scenario=scenario)

        naive_srisps = NaiveSRISPSHighDimension(data_creator=data_creator, epochs=epochs, learning_rate=learning_rate)
        coefficients = naive_srisps.estimate_naive_regression()
        for i, coef in enumerate(coefficients[0]):
            results['method'].append('Naive Regression')
            results['coefficient'].append(f'coef_{i}')
            results['value'].append(coef)

        coefficients = naive_srisps.estimate_sps()
        for i, coef in enumerate(coefficients[0]):
            results['method'].append('SPS')
            results['coefficient'].append(f'coef_{i}')
            results['value'].append(coef)

        coefficients = naive_srisps.estimate_sri()
        for i, coef in enumerate(coefficients[0]):
            results['method'].append('SRI')
            results['coefficient'].append(f'coef_{i}')
            results['value'].append(coef)
        # NN model
        coefficients_sri, coefficients_sps = naive_srisps.estimating_sri_sps_with_nn()
        for i, coef in enumerate(coefficients_sri[0]):
            results['method'].append('SRI with NN')
            results['coefficient'].append(f'coef_{i}')
            results['value'].append(coef)
            if i == 0:
                print(f'coef_{i} sri: {coef}')
        for i, coef in enumerate(coefficients_sps[0]):
            results['method'].append('SPS with NN')
            results['coefficient'].append(f'coef_{i}')
            results['value'].append(coef)
            if i == 0:
                print(f'coef_{i} sps: {coef}')

    results_df = pd.DataFrame(results)
    results_df.to_pickle(f'high_dim_sim_results_{num_simulations}_{beta_u}_{beta_3}_{scenario}_{non_null_iv}.pkl')
    return results_df

if __name__ == '__main__':
    lr = 0.001
    num_simulations: int = 1
    beta_2: float = 1
    beta_1: float = -1
    n: int = 20000
    p: int = 1000
    gamma_iv : float = 0.05
    std = gamma_iv/2
    gamma_u : float = 1
    epochs : int = 1000
    l1_lambda = 0
    for scenario in [2]:
        for non_null_iv in [10, 100, 500, 1000]:
            for beta_u_ in [1]:
                for beta_3_ in [0.5]:
                    res = run_high_dimension_genetic_simulation(num_simulations=num_simulations,
                                                                beta_u=beta_u_,
                                                                beta_3=beta_3_,
                                                                scenario=scenario,
                                                                p=p,
                                                                n=n,
                                                                beta_1=beta_1,
                                                                beta_2=beta_2,
                                                                gamma_u=gamma_u,
                                                                non_null_iv = non_null_iv,
                                                                learning_rate=lr,
                                                                epochs = epochs)
                    plot_boxplot(res, y_line=beta_1)