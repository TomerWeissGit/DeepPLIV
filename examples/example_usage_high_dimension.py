import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from core.trainer import DeepPLIV
from utils.helpers import plot_boxplot
from typing import Literal
from sklearn.preprocessing import StandardScaler

class SimDataCreatorHighDimension:
    def __init__(self,
                 n: int,
                 beta_1: float,
                 beta_2: float,
                 beta_3: float,
                 beta_u: float,
                 gamma_u: float,
                 gwas_threshold: float = None,
                 m: int = 1000000,
                 scenario: int = 1,
                 only_first_stage: bool = True,
                 sigma_iv: float = np.sqrt(10e-5)):
        """
        Initialize the SimDataCreatorHighDimension class.

        :param n: int, number of samples.
        :param beta_1: float, coefficient for the endogenous variable.
        :param beta_2: float, coefficient for the exogenous variable.
        :param beta_3: float, coefficient for the interaction term between the endogenous and exogenous variables.
        :param beta_u: float, coefficient for the confounding variable.
        :param gamma_u: float, coefficient for the confounding variable in the first stage.
        :param gwas_threshold: float, threshold for the GWAS.
        :param scenario: int, scenario to consider.
        :param only_first_stage: bool, whether to generate only the first stage data.
        """
        self.n_x_1 = self.n_x_2 = self.n_y = n // 2
        self.u = np.random.normal(0, 1, n)
        self.m = m
        self.beta_1, self.beta_2, self.beta_3, self.beta_u, self.gamma_u = beta_1, beta_2, beta_3, beta_u, gamma_u
        self.gamma_iv = np.array([])
        self.gwas_threshold = 5e-6 if gwas_threshold is None else gwas_threshold
        self.sigma_iv = sigma_iv
        self._generate_gamma_iv()
        self.g_iv = np.random.binomial(2, 0.3, size=(n, self.gamma_iv.shape[0])) / 2
        self.g_iv_1, self.g_iv_2 = self.g_iv[:self.n_x_1], self.g_iv[self.n_x_1:]

        self.x_1, self.x_2 = self._generate_x_df(scenario)
        self.y = self._generate_y_df_linear() if only_first_stage else self._generate_y_df_non_linear(scenario)

    def _generate_gamma_iv(self):
        gamma_iv = np.random.normal(0, self.sigma_iv, self.m)
        gamma_iv_for_gwas = np.random.normal(loc=gamma_iv, scale=np.sqrt(1 / self.n_x_1), size=self.m)
        p_values = 2 * (1 - norm.cdf(np.sqrt(self.n_x_1) * np.abs(gamma_iv_for_gwas)))
        self.gamma_iv = gamma_iv[p_values < self.gwas_threshold]

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
        # elif scenario == 2:
        #     x_1 = (np.abs(self.g_iv_1 - 0.75) * 2) @ self.gamma_iv + self.gamma_u * self.u[:self.n_x_1] + error_1
        #     x_2 = (np.abs(self.g_iv_2 - 0.75) * 2) @ self.gamma_iv + self.gamma_u * self.u[self.n_x_1:] + error_2
        # elif scenario == 3:
        #     x_1 = (self.g_iv_1 ** 2) @ self.gamma_iv + self.gamma_u * self.u[:self.n_x_1] + error_1
        #     x_2 = (self.g_iv_2 ** 2) @ self.gamma_iv + self.gamma_u * self.u[self.n_x_1:] + error_2
        # elif scenario == 4:
        #     x_1 = (self.g_iv_1==0.5).astype(float) @ self.gamma_iv + self.gamma_u * self.u[:self.n_x_1] + error_1
        #     x_2 = (self.g_iv_2==0.5).astype(float) @ self.gamma_iv + self.gamma_u * self.u[self.n_x_1:] + error_2
        # elif scenario == 5:
        #     x_1 = (np.exp(self.g_iv_1)).astype(float) @ self.gamma_iv + self.gamma_u * self.u[:self.n_x_1] + error_1
        #     x_2 = (np.exp(self.g_iv_2)).astype(float) @ self.gamma_iv + self.gamma_u * self.u[self.n_x_1:] + error_2
        elif scenario == 2:
            # Define the number of interactions to use
            num_interactions = self.g_iv_1.shape[1] // 2
            # Randomly select pairs of indices
            interaction_indices = np.random.choice(self.g_iv_1.shape[1], size=(num_interactions, 2), replace=False)
            def aux_interaction_effect(g_iv):
                interaction_matrix = g_iv[:, interaction_indices[:, 0]] * g_iv[:, interaction_indices[:, 1]]
                interaction_effects = interaction_matrix @ ((self.gamma_iv[interaction_indices[:, 0]] +
                                                             self.gamma_iv[interaction_indices[:, 1]]) / 2)
                return interaction_effects
            # Create interaction matrix for g_iv_1
            interaction_effects_1 = aux_interaction_effect(self.g_iv_1)
            # Create interaction matrix for g_iv_2
            interaction_effects_2 = aux_interaction_effect(self.g_iv_2)
            # Compute x_1 and x_2
            x_1 = self.g_iv_1 @ self.gamma_iv + interaction_effects_1 + self.gamma_u * self.u[:self.n_x_1] + error_1
            x_2 = self.g_iv_2 @ self.gamma_iv + interaction_effects_2 + self.gamma_u * self.u[self.n_x_1:] + error_2
            return x_1, x_2
        elif scenario == 3:
            # Define the number of interactions to use
            num_interactions = self.g_iv_1.shape[1] // 2
            # Randomly select pairs of indices
            interaction_indices = np.random.choice(self.g_iv_1.shape[1], size=(num_interactions, 2), replace=False)

            def aux_interaction_effect(g_iv):
                interaction_matrix = g_iv[:, interaction_indices[:, 0]] * g_iv[:, interaction_indices[:, 1]]
                interaction_effects = interaction_matrix @ ((self.gamma_iv[interaction_indices[:, 0]] +
                                                             self.gamma_iv[interaction_indices[:, 1]]) / 2)
                return interaction_effects

            # Create interaction matrix for g_iv_1
            interaction_effects_1 = aux_interaction_effect(self.g_iv_1)
            # Create interaction matrix for g_iv_2
            interaction_effects_2 = aux_interaction_effect(self.g_iv_2)
            # Compute x_1 and x_2
            x_1 =  interaction_effects_1 + self.gamma_u * self.u[:self.n_x_1] + error_1
            x_2 =  interaction_effects_2 + self.gamma_u * self.u[self.n_x_1:] + error_2
            return x_1, x_2
        else:
            raise ValueError('Scenario not implemented')
        return self._standardize(x_1, x_2)

    def _generate_y_df_linear(self):
        self.g = np.random.binomial(2, 0.3, size=self.n_y)
        y_base = self.beta_2 * self.g + self.beta_u * self.u[self.n_x_1:] + np.random.normal(0, 1, self.n_y)
        return self.beta_1 * self.x_2 + y_base + self.beta_3 * (self.g * self.x_2)

    def _generate_y_df_non_linear(self, scenario):
        self.g = np.random.binomial(2, 0.3, size=(self.n_y, 1))
        beta_g = np.random.normal(0, 1, 1)
        y_base = self.beta_u * self.u[self.n_x_1:] + np.random.normal(0, 1, self.n_y)
        if scenario == 1:
            return self.beta_1 * self.x_2 + self.g @ beta_g + y_base
        elif scenario == 2:
            return self.beta_1 * self.x_2 + ((self.g>0).astype(int) @ beta_g) + y_base
        elif scenario == 3:
            return self.beta_1 * self.x_2 + (self.g**2) @ beta_g + y_base
        elif scenario == 4:
            return self.beta_1 * self.x_2 + (np.exp(self.g)) @ beta_g + y_base
        elif scenario == 5:
            return (self.beta_1 * self.x_2 + (self.g[:, 0] * self.g[:, 1] + self.g[:, 2] * self.g[:, 3])
                    + self.g @ beta_g + y_base)

    def get_data(self):
        return self.x_1, self.x_2, self.y, self.g_iv_1, self.g_iv_2, self.g


class NaiveSRISPSHighDimension:
    def __init__(self,
                 data_creator: SimDataCreatorHighDimension,
                 epochs: int = 1000,
                 learning_rate: float = 0.001,
                 first_dropout: float = 0.6,
                 second_dropout: float = 0.2):
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
        self.first_dropout = first_dropout
        self.second_dropout = second_dropout
    def estimate_naive_regression(self):
        """
        Estimate the Naive SR-IV model using a simple linear regression model.
        :return: np.array, the coefficients of the Naive SR-IV model.
        """
        if len(self.g.shape) > 1:
            x = np.concatenate((self.x_2.reshape(-1, 1), self.g, (self.g * self.x_2[:, None])), axis=1)
        else:
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
                                                  first_dropout=self.first_dropout,
                                                  second_dropout=self.second_dropout,
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
                                          scenario: Literal[1, 2] =2,
                                          n: int = 20000,
                                          gamma_u: float = 1.0,
                                          learning_rate: float = 0.01,
                                          epochs: int = 2000,
                                          gwas_threshold: float = None,
                                          k: int = 5,
                                          m: int = 1000000,
                                          sigma_iv: float = np.sqrt(10e-5),
                                          linear_second_stage: bool = True,
                                          first_dropout: float = 0.6,
                                          second_dropout: float = 0.2):
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
    :param m: parameter to control the number of samples for the genetic instrument.
    :param gamma_u: parameter to control the coefficient for the confounding variable.
    :param learning_rate: parameter to control the learning rate for the neural network model.
    :param epochs: parameter to control the number of epochs for training the neural network model.
    :param gwas_threshold: parameter to control the threshold for the GWAS.
    :param k: parameter to control the number of trainings for the neural network model.
    :param linear_second_stage: parameter to control whether to generate only the first part as non-linear.
    :return: pd.DataFrame, the results of the simulation.
    """
    results = {
        'method': [],
        'coefficient': [],
        'value': []
    }

    for _ in range(num_simulations):
        data_creator = SimDataCreatorHighDimension(n=n, beta_1=beta_1, beta_2=beta_2, beta_3=beta_3, beta_u=beta_u,
                                                   gamma_u=gamma_u, scenario=scenario, m=m, sigma_iv=sigma_iv,
                                                   gwas_threshold=gwas_threshold, only_first_stage=linear_second_stage)

        naive_srisps = NaiveSRISPSHighDimension(data_creator=data_creator, epochs=epochs, learning_rate=learning_rate,
                                                first_dropout=first_dropout, second_dropout=second_dropout)

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
    # results_df.to_pickle(f'high_dim_sim/{n}_{num_simulations}_{beta_u}_{beta_3}_{scenario}_with_{k}_trainings_dropout_{first_dropout}.pkl')
    return results_df



if __name__ == '__main__':
    num_simulations: int = 10
    # This is the simulation for the first only the first part being non-linear
    beta_2: float = 1
    beta_1: float = 0.5
    n: int =  5000
    k: int = 1
    m: int = 300
    lr: float = 0.0000001
    gamma_u : float = 0.5
    second_dropout: float = 0.1
    gwas_threshold: float = 5e-8
    beta_3_ = 0.5
    beta_u_ = 1
    sigma_iv = np.sqrt(5e-3)
    for scenario in [3]:
        for n in [200000]:
            first_dropout = 1000/(1000+n)
            epochs: int = int((3 * 10 ** 6) / n)
            print(f'scenario: {scenario}, n: {n}, first_dropout: {first_dropout}')
            res = run_high_dimension_genetic_simulation(num_simulations=num_simulations,
                                                        beta_u=beta_u_,
                                                        beta_3=beta_3_,
                                                        scenario=scenario,
                                                        n=n,
                                                        m=m,
                                                        beta_1=beta_1,
                                                        beta_2=beta_2,
                                                        gamma_u=gamma_u,
                                                        learning_rate=lr,
                                                        epochs = epochs,
                                                        gwas_threshold=gwas_threshold,
                                                        k = k,
                                                        linear_second_stage=True,
                                                        first_dropout=first_dropout,
                                                        second_dropout=second_dropout,
                                                        sigma_iv=sigma_iv)
            # res = pd.read_pickle(f'high_dim_sim/high_dropout_{num_simulations}_{beta_u_}_{beta_3_}_{scenario}_with_{k}_trainings.pkl')
            plot_boxplot(res, y_line=beta_1)





