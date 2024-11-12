import numpy as np
from typing_extensions import Literal
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
            binomial_part = int((z_b_1 < (-0.5)) | (z_b_2 < (-0.5))) - int((z_b_1 > (-0.5)) | (z_b_2 > (-0.5)))
            return sum_coefficients + binomial_part + z_b_3 * z_b_4 + gamma_u_eta


if __name__ == '__main__':
    sim_data = SimDataCreator()
    print(sim_data.y)
    print(sim_data.v)