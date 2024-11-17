import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from typing_extensions import Literal

from core.trainer import DeepPLIV
from utils.helpers import spinner_decorator


class SimDataCreator:
    def __init__(self, p: int = 20,
                 n: int = 10000,
                 beta_1: float = 1,
                 beta_u: float = 5,
                 gamma_u: float = 3,
                 std_epsilon: float = 5,
                 std_eta: float = 3,
                 scenario: Literal[1, 2, 3, 4, 5] = 1):
        """
        Create a simulated data set for the partially linear IV model.
        The data is generated according to the following model:
        y = beta_1 * v + beta_u(u) + f(x) + epsilon
        v = g(z) + gamma_u(u) + eta
        where:
        - y is the dependent variable
        - v is the endogenous variable
        - z is the instrumental variable
        - u is the confounding variable
        - epsilon is the error term
        - eta is the error term for the endogenous variable
        - f and g are unknown non-linear functions.
        The data is generated according to the following steps:
        1. Generate the z matrix according to the _get_z_matrix method.
        2. Generate the u vector from a normal distribution.
        3. Generate the v vector according to the get_v_vector method.
        4. Generate the y vector according to the model.
        :param p: int, the number of features in the z matrix.
        :param n: int, the number of samples.
        :param beta_1: float, the coefficient for the endogenous variable.
        :param beta_u: float, the coefficient for the confounding variable.
        :param gamma_u: float, the coefficient for the confounding variable in the endogenous variable equation.
        :param std_epsilon: float, the standard deviation of the error term.
        :param std_eta: float, the standard deviation of the error term for the endogenous variable.
        :param scenario: int, the scenario for the function g.
        """
        if scenario not in [1, 2, 3, 4, 5]:
            raise ValueError("Scenario must be 1, 2, 3, 4 or 5")
        self.scenario = scenario
        self.p = p
        self.n = n
        self.beta_1 = beta_1
        self.beta_u = beta_u
        self.gamma_u = gamma_u
        self.u: np.array = np.random.normal(0, 1, self.n)
        z, z_coefficients = self._get_z_matrix()
        self.z: np.array = z
        self.z_coefficients: np.array = z_coefficients
        self.epsilon: np.array = np.random.normal(0, std_epsilon, n)
        self.eta: np.array = np.random.normal(0, std_eta, n)
        self.v: np.array = self.get_v_vector()
        self.y: np.array = self.get_y()

    @spinner_decorator("Generating y vector")
    def get_y(self) -> np.array:
        return self.beta_1 * self.v + self.beta_u * self.u + self.epsilon

    @spinner_decorator("Generating z matrix")
    def _get_z_matrix(self) -> (np.array, np.array):
        """
        Create a matrix of z vectors for the instrumental variable.
        The z vectors are generated according to the following model:
            - The first 20% of the data are from a multivariate normal distribution.
            - The second 20% of the data are binomial.
            - The rest of the data are generated according to a multinomial distribution.
            - The first 40% of the data are multiplied by 0.2.
            - The rest of the data are multiplied by a coefficient from a multivariate normal distribution,
             with mean 0.2 and covariance matrix 0.01 * sigma.
        the shape of the matrix is determined by self.n X self.p
        :return: z_coefficients_matrix, z_all_matrix where z_coefficients_matrix is the matrix of coefficients for the
        z vectors and z_all_matrix is the matrix of the z vectors.
        """

        def _aux_get_z_vector():
            p = self.p
            # sigma is a p x p matrix with sigma = exp(-|j-j'|, 1<j,j'<p)
            sigma: np.array = np.fromfunction(lambda i, j: np.exp(-np.abs(i - j)), (p, p))
            # z is a multivariate normal with mean 0 and covariance matrix sigma
            z = np.random.multivariate_normal([0] * p, sigma)
            # first 20 percent of the data are from the multivariate normal distribution
            z_first_20_percent = z[:int(p * 0.2)].tolist()
            # second 20 percent of the data are 1 if the data is positive and 0 otherwise, i.e. binomial.
            z_second_20_percent = (z[int(p * 0.2):int(p * 0.4)] > 0).astype(int).tolist()
            # rest is 2 if the data is greater than 0.5, 1 if the data is between -0.5 and 0.5 and 0 otherwise.
            z_rest = ((z[int(p * 0.4):] > -0.5).astype(int) + (z[int(p * 0.4):] > 0.5).astype(int)).tolist()
            # combining all together
            z_all = z_first_20_percent + z_second_20_percent + z_rest
            # first 40 percent of the data are multiplied by 0.2
            coefficients_first_40_percent = [z * 0.2 for z in z_first_20_percent + z_second_20_percent]
            # other 60 percent of the data coefficient are from the multivariate normal distribution with mean 0.2 and
            # covariance matrix 0.01 * sigma
            multivariate_normal_coefficients = np.random.multivariate_normal([0.2] * p, sigma * 0.01)[-len(z_rest):]
            coefficients_other_60_percent = (np.array(z_rest) * multivariate_normal_coefficients).tolist()

            z_coefficients = coefficients_first_40_percent + coefficients_other_60_percent
            return z_coefficients, z_all

        z_coefficients_matrix = []
        z_all_matrix = []
        for _ in range(self.n):
            z_coefficients_vector, z_all_vector = _aux_get_z_vector()
            z_coefficients_matrix.append(z_coefficients_vector)
            z_all_matrix.append(z_all_vector)
        return np.array(z_coefficients_matrix), np.array(z_all_matrix)

    @spinner_decorator("Generating v vector")
    def get_v_vector(self) -> np.array:
        """
        Create a vector of v values for the endogenous variable.
        The v values are generated according to the following model:
        v = g(z) + gamma_u(u) + eta
        where:
        - v is the endogenous variable
        - z is the instrumental variable
        - u is the confounding variable
        - eta is the error term for the endogenous variable
        - g is an unknown non-linear function.
        there are 5 different scenarios for the function g:
        1. g(z) = sum(z_i) -> this is the only linear function of z.
        2. g(z) = sum(z_i) + z_1^2 + z_2^2
        3. g(z) = sum(z_i) + z_3 * z_4
        4. g(z) = sum(z_i) + z_1^2 + z_2^2 + z_3 * z_4
        5. g(z) = sum(z_i) + binomial(z_1, z_2) + z_3 * z_4
        where z_1, z_2, z_3, z_4 are the first, second, third and fourth elements of z.
        :return: np.array, the vector of v values.
        """
        sum_coefficients = self.z_coefficients.sum(axis=1)
        z_b_1 = self.z_coefficients[:, 0]
        z_b_2 = self.z_coefficients[:, 1]
        z_b_3 = self.z_coefficients[:, 2]
        z_b_4 = self.z_coefficients[:, 3]
        gamma_u_eta = self.gamma_u * self.u + self.eta
        if self.scenario == 1:
            return sum_coefficients + gamma_u_eta
        if self.scenario == 2:
            return sum_coefficients + z_b_1 ** 2 + z_b_2 ** 2 + gamma_u_eta
        if self.scenario == 3:
            return sum_coefficients + z_b_3 * z_b_4 + gamma_u_eta
        if self.scenario == 4:
            return sum_coefficients + z_b_1 ** 2 + z_b_2 ** 2 + z_b_3 * z_b_4 + gamma_u_eta
        if self.scenario == 5:
            binomial_part = (((z_b_1 < (-0.5)) | (z_b_2 < (-0.5))).astype(int)
                             - ((z_b_1 > (-0.5)) | (z_b_2 > (-0.5))).astype(int))
            return sum_coefficients + binomial_part + z_b_3 * z_b_4 + gamma_u_eta


def run_simulation(scenario, p=20, n=2000, beta_1=0.5, beta_u=1, gamma_u=1, std_epsilon=1, std_eta=1,
                   first_stage_epochs=3000, first_stage_learning_rate=0.01, second_stage_epochs=3000,
                   second_stage_learning_rate=0.01):
    deep_pliv = DeepPLIV()
    data = SimDataCreator(p, n, beta_1, beta_u, gamma_u, std_epsilon, std_eta, scenario)
    v, z = data.v, data.z

    deep_pliv._fit_first_stage(v, z, first_stage_epochs, first_stage_learning_rate)
    first_stage_2sls_model = LinearRegression()
    first_stage_2sls_model.fit(z, v)

    data_2 = SimDataCreator(p, n, beta_1, beta_u, gamma_u, std_epsilon, std_eta, scenario)
    z_second = data_2.z

    z_predicted_linear_regression = first_stage_2sls_model.predict(z_second)
    v_predicted = deep_pliv._predict_first_stage(z_second)

    v_pred_error_linear_regression = z_predicted_linear_regression - data_2.v
    v_pred_error = v_predicted - data_2.v

    x = np.concatenate((v_pred_error, np.ones((n, 1))), axis=1)
    y = data_2.y
    v = data_2.v.reshape(-1, 1)

    second_stage_linear_model = LinearRegression()
    x_linear_regression = np.concatenate(((v_pred_error_linear_regression - data_2.v).reshape(-1, 1), np.ones((n, 1))),
                                         axis=1)
    x_all = np.concatenate((x_linear_regression, v), axis=1)
    second_stage_linear_model.fit(x_all, y)

    deep_pliv._fit_second_stage(v, x, y, second_stage_epochs, second_stage_learning_rate)
    y_predicted = deep_pliv._predict_second_stage(v, x)
    predicted_error_deep = np.sqrt(((y_predicted - y) ** 2).mean())

    return {
        'scenario': scenario,
        'predicted_error_linear': np.sqrt(((second_stage_linear_model.predict(x_all) - y) ** 2).mean()),
        'predicted_error_deep': predicted_error_deep,
        'linear_coefficients': second_stage_linear_model.coef_[-1],
        'deep_coefficients': deep_pliv.get_v_predicted_coefficient()
    }


def run_scenario_simulations(scenario, num_simulations):
    scenario_results = []
    for i in range(num_simulations):
        result = run_simulation(scenario)
        scenario_results.append(result)
        save_to_pickle(scenario_results, f'scenario_{scenario}_{num_simulations}_results.pkl')
        print(f"Iteration {i + 1}/{num_simulations} for scenario {scenario}")
    return {
        'scenario': scenario,
        'average_predicted_error_linear': np.mean([r['predicted_error_linear'] for r in scenario_results]),
        'average_predicted_error_deep': np.mean([r['predicted_error_deep'] for r in scenario_results]),
        'average_linear_coefficients': np.mean([r['linear_coefficients'] for r in scenario_results], axis=0),
        'average_deep_coefficients': np.mean([r['deep_coefficients'] for r in scenario_results]),
        'std_linear_coefficients': np.std([r['linear_coefficients'] for r in scenario_results], axis=0),
        'std_deep_coefficients': np.std([r['deep_coefficients'] for r in scenario_results])
    }


def monte_carlo_simulation(num_simulations=100):
    with mp.Pool(processes=5) as pool:
        results = pool.starmap(run_scenario_simulations, [(scenario, num_simulations) for scenario in range(1, 6)])
    return results


def save_to_pickle(lst: list, name: str = 'results.pkl'):
    import pickle
    with open(name, 'wb') as f:
        pickle.dump(lst, f)


# if __name__ == '__main__':
#     num_simulations = 1
#     results = monte_carlo_simulation(num_simulations)
#     save_to_pickle(results, '1_results_n_2000_20_beta1_05.pkl')
#     for result in results:
#         print(f"Scenario {result['scenario']}:")
#         print(f"  Average Predicted Error (Linear): {result['average_predicted_error_linear']}")
#         print(f"  Average Predicted Error (Deep): {result['average_predicted_error_deep']}")
#         print(f"  Average Linear Coefficients: {result['average_linear_coefficients']}")
#         print(f"  Average Deep Coefficients: {result['average_deep_coefficients']}")
#         print(f"  Std Linear Coefficients: {result['std_linear_coefficients']}")
#         print(f"  Std Deep Coefficients: {result['std_deep_coefficients']}")

# with open('results.pkl', 'rb') as f:
#     results = pickle.load(f)

# Y = \beta_0 + \beta_1 * X + \beta_2 * G + \beta_3 * XG + \beta_z * Z + \beta_u * U + \epsilon_Y
# X = \gamma_0 + \gamma_iv * G_iv + \gamma_z * Z + \gamma_u * U + \epsilon_X
class NaiveSRISPS:

    def __init__(self, beta_u: float = 0, beta_3: float = 0, gamma_u: float = 1, dependent: int = 0):
        """
        Create a NaiveSRISPS object. The object is used to store the coefficients of the Naive SR-IV model. The model is
        defined as follows:
        Y = \beta_0 + \beta_1 * X + \beta_2 * G + \beta_3 * XG + \beta_z * Z + \beta_u * U + \epsilon_Y
        X = \gamma_0 + \gamma_iv * G_iv + \gamma_z * Z + \gamma_u * U + \epsilon_X
        where:
        - Y is the dependent variable
        - X is the endogenous variable
        - G is the exogenous variable which imitates genes with an interaction effect on Y with X.
        - G_iv is the instrument variable for the endogenous variable.
        - Z is the exogenous variable with no interaction effect on Y with X.
        - U is the confounding variable.
        - \epsilon_Y and \epsilon_X are the error terms for Y and X.
        :param beta_u: float, the coefficient for the confounding variable in the Y equation.
        :param beta_3: float, the coefficient for the interaction term between Z variable and X variable in the Y equation.
        :param gamma_u: float, the coefficient for the confounding variable in the X equation.
        :param dependent: int, if there is dependency between g and g_iv or not.
        """
        # setting the random variables for the model.
        self.u = np.random.normal(0, 1, 10000)
        self.z = np.random.normal(0, 1, 10000)
        self.epsilon_x = np.random.normal(0, 1, 10000)
        self.epsilon_y = np.random.normal(0, 1, 10000)
        self.g_iv = np.random.normal(0, 1, 10000)
        # setting the coefficients for the model.
        self.beta_0 = 0
        self.beta_1 = 1
        self.beta_2 = 0.5
        self.beta_3 = beta_3
        self.beta_z = 0.5
        self.beta_u = beta_u
        self.gamma_0 = 0
        self.gamma_iv = 0.5
        self.gamma_z = 0.5
        self.gamma_u = gamma_u
        # calculating g variable
        self.g = np.array([np.random.binomial(2, 0.3 + (0.3 * (iv > 0) * dependent), 1)[0] for iv in self.g_iv])
        # calculating the X variable
        self.x = (self.gamma_0
                  + self.gamma_iv * self.g_iv
                  + self.gamma_z * self.z
                  + self.gamma_u * self.u
                  + self.epsilon_x)
        # calculating the Y variable
        self.y = (self.beta_0
                  + self.beta_1 * self.x
                  + self.beta_2 * self.g
                  + self.beta_3 * self.x * self.g
                  + self.beta_z * self.z
                  + beta_u * self.u
                  + self.epsilon_y)

    def estimate_naive_regression(self):
        """
        Estimate the Naive SR-IV model using a simple linear regression model.
        :return: np.array, the coefficients of the Naive SR-IV model.
        """
        x = np.concatenate((self.x.reshape(-1, 1),
                            self.g.reshape(-1, 1),
                            self.x.reshape(-1, 1) * self.g.reshape(-1, 1),
                            self.z.reshape(-1, 1)),
                           axis=1)
        y = self.y
        model = LinearRegression()
        model.fit(x, y)
        return model.coef_

    def estimate_naive_regression(self):
        """
        Estimate the Naive SR-IV model using a simple linear regression model.
        :return: np.array, the coefficients of the Naive SR-IV model.
        """

        x = np.concatenate((self.x.reshape(-1, 1),
                            self.g.reshape(-1, 1),
                            self.x.reshape(-1, 1) * self.g.reshape(-1, 1),
                            self.z.reshape(-1, 1)),
                           axis=1)
        y = self.y
        model = LinearRegression()
        model.fit(x, y)
        return model.coef_

    def get_first_and_second_stage_x_data(self):
        """
        Get the data for the first stage of the 2SLS model.
        :return: np.array, the data for the first stage of the 2SLS model.
        """
        x_first_stage = self.x[:5000]
        g_iv_first_stage = self.g_iv[:5000]
        z_first_stage = self.z[:5000]
        vars_first_stage = np.concatenate((g_iv_first_stage.reshape(-1, 1), z_first_stage.reshape(-1, 1)), axis=1)

        x_second_stage = self.x[5000:]
        g_iv_second_stage = self.g_iv[5000:]
        z_second_stage = self.z[5000:]
        vars_second_stage = np.concatenate((g_iv_second_stage.reshape(-1, 1), z_second_stage.reshape(-1, 1)), axis=1)
        return vars_first_stage, x_first_stage, vars_second_stage, x_second_stage

    def estimate_sps(self):
        """
        Estimate the 2SLS model using a simple linear regression model. - SPS
        :return: np.array, the coefficients of the Naive SR-IV model.
        """

        vars_first_stage, x_first_stage, vars_second_stage, x_second_stage = self.get_first_and_second_stage_x_data()
        model = LinearRegression()
        model.fit(vars_first_stage, x_first_stage)
        x_predicted = model.predict(vars_second_stage)
        x = np.concatenate((x_predicted.reshape(-1, 1),
                            self.g[5000:].reshape(-1, 1),
                            x_predicted.reshape(-1, 1) * self.g[5000:].reshape(-1, 1),
                            self.z[5000:].reshape(-1, 1)),
                           axis=1)
        y = self.y[5000:]
        model = LinearRegression()
        model.fit(x, y)
        return model.coef_

    def estimate_sri(self):
        """
        Estimate the 2SLS model using a simple linear regression model - SRI
        :return: np.array, the coefficients of the Naive SR-IV model.
        """

        vars_first_stage, x_first_stage, vars_second_stage, x_second_stage = self.get_first_and_second_stage_x_data()
        model = LinearRegression()
        model.fit(vars_first_stage, x_first_stage)
        x_predicted = model.predict(vars_second_stage)
        x_error = x_second_stage.reshape(-1,1) - x_predicted

        x = np.concatenate((x_second_stage.reshape(-1, 1),
                            self.g[5000:].reshape(-1, 1),
                            x_second_stage.reshape(-1, 1) * self.g[5000:].reshape(-1, 1),
                            self.z[5000:].reshape(-1, 1),
                            x_error),
                           axis=1)
        y = self.y[5000:]
        model = LinearRegression()
        model.fit(x, y)
        return model.coef_
    def estimating_sri_with_nn(self):
        """
        Estimate the 2SLS model using a simple neural network model. - SPS, this class is going to use the DeepPLIV
        model, which is a deep learning model for the 2SLS model.
        :return: np.array, the coefficients of the Naive SR-IV model.
        """

        vars_first_stage, x_first_stage, vars_second_stage, x_second_stage = self.get_first_and_second_stage_x_data()
        model = DeepPLIV()
        first_stage_model = model._fit_first_stage(x_first_stage, vars_first_stage, 2000, 0.001)
        x_predicted = first_stage_model.predict(vars_second_stage)
        x_error = x_second_stage.reshape(-1,1) - x_predicted

        x = np.concatenate((x_second_stage.reshape(-1, 1),
                            self.g[5000:].reshape(-1, 1),
                            x_second_stage.reshape(-1, 1) * self.g[5000:].reshape(-1, 1),
                            self.z[5000:].reshape(-1, 1),
                            x_error),
                           axis=1)
        y = self.y[5000:]
        model = LinearRegression()
        model.fit(x, y)
        return model.coef_


def run_simulation(num_simulations=300):
    naive_srisps = NaiveSRISPS(beta_u=3, beta_3=0.5, gamma_u=1, dependent=0)
    results = {
        'method': [],
        'coefficient': [],
        'value': []
    }

    for _ in range(num_simulations):
        coefficients = naive_srisps.estimate_naive_regression()
        for i, coef in enumerate(coefficients):
            results['method'].append('Naive Regression')
            results['coefficient'].append(f'coef_{i}')
            results['value'].append(coef)

        coefficients = naive_srisps.estimate_sps()
        for i, coef in enumerate(coefficients):
            results['method'].append('SPS')
            results['coefficient'].append(f'coef_{i}')
            results['value'].append(coef)

        coefficients = naive_srisps.estimate_sri()
        for i, coef in enumerate(coefficients):
            results['method'].append('SRI')
            results['coefficient'].append(f'coef_{i}')
            results['value'].append(coef)

        coefficients = naive_srisps.estimating_sps_with_nn()
        for i, coef in enumerate(coefficients):
            results['method'].append('SPS with NN')
            results['coefficient'].append(f'coef_{i}')
            results['value'].append(coef)
  #  save_to_pickle(results, 'sim_results_with_nn_model.pkl')
    return pd.DataFrame(results)


def plot_boxplot(df):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='coefficient', y='value', hue='method', data=df)
    plt.title('Coefficient Distribution by Method')
    plt.show()


if __name__ == '__main__':
    df = run_simulation(num_simulations=1)
    plot_boxplot(df)
