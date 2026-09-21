import itertools

import numpy as np
from sklearn.preprocessing import StandardScaler

from scipy.stats import norm
import random

global CHOSEN_INTERACTION_PAIRS

import numpy as np
import itertools

class SimDataCreatorHighDimension:
    def __init__(self, n, m, betas, gamma_u, p_thr, interaction=False, chosen_pairs=None, interaction_effects=None):
        self.n_x = int(n // 2)
        self.n_y = int(n // 2)
        self.m = int(m)
        self.betas = betas  # (beta1, beta2, beta_u)
        self.gamma_u = gamma_u
        self.p_thr = p_thr
        self.interaction = interaction
        self.chosen_pairs = chosen_pairs
        self.interaction_effects = interaction_effects

        self.x_1, self.x_2, self.z_1, self.z_2, self.g, self.y = self._generate_data()

    def _generate_data(self):
        beta1, beta2, beta_u = self.betas
        gamma_u = self.gamma_u

        sigma_iv = np.sqrt(5e-3)
        gamma_ivs = np.random.normal(0, sigma_iv, self.m)
        gamma_ivs_for_gwas = np.random.normal(loc=gamma_ivs, scale=np.sqrt(1 / self.n_x), size=self.m)

        p_values = 2 * (1 - norm.cdf(np.sqrt(self.n_x) * np.abs(gamma_ivs_for_gwas)))
        mask = (p_values < self.p_thr)
        gamma_ivs_filtered = gamma_ivs[mask][:500] if len(gamma_ivs[mask]) > 500 else gamma_ivs[mask]
        number_of_ivs = len(gamma_ivs_filtered)

        g_iv_n_x = np.random.binomial(n=2, p=0.3, size=(self.n_x, number_of_ivs))
        g_iv_n_y = np.random.binomial(n=2, p=0.3, size=(self.n_y, number_of_ivs))
        u_n_x = np.random.normal(0, 1, self.n_x)
        u_n_y = np.random.normal(0, 1, self.n_y)

        if self.interaction:
            chosen_pairs = self.chosen_pairs
            interaction_effects = self.interaction_effects

            x_n_x = self.generate_x_with_interactions(g_iv_n_x, chosen_pairs, interaction_effects) + gamma_u * u_n_x
            x_n_y = self.generate_x_with_interactions(g_iv_n_y, chosen_pairs, interaction_effects) + gamma_u * u_n_y
        else:
            x_n_x = g_iv_n_x @ gamma_ivs_filtered + gamma_u * u_n_x
            x_n_y = g_iv_n_y @ gamma_ivs_filtered + gamma_u * u_n_y

        g = np.random.binomial(2, 0.25, self.n_y)
        g = g - np.mean(g)

        if self.interaction:
            y = (beta1 * x_n_y + (np.exp(-beta2 * g + 5)) ** 1.2 * ((g ** 2) ** 0.5) + beta_u * u_n_y
                 + np.random.normal(0, 1, self.n_y))
        else:
            y = beta1 * x_n_y + beta2 * g + beta_u * u_n_y + np.random.normal(0, 1, self.n_y)

        return x_n_x, x_n_y, g_iv_n_x, g_iv_n_y, g, y

    def get_data(self):
        return self.x_1, self.x_2, self.z_1, self.z_2, self.g, self.y

    @staticmethod
    def generate_x_with_interactions(g_iv, chosen_pairs, interaction_effects):
        x_contrib = np.zeros(g_iv.shape[0])
        for (i, j), effect in zip(chosen_pairs, interaction_effects):
            x_contrib += effect * (g_iv[:, i] * g_iv[:, j])
        return x_contrib


class SimDataCreatorDeepIV:
    def __init__(self,
                 n: int,
                 beta_1: float,
                 rho: float = 0.1):
        """
        Initialize the SimDataCreatorDeepIV class.

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
        y = y + self.beta_1 * self.x_2 + self.e# scale the data
        return y, s


    def get_data(self):
        return self.x_1, self.x_2, self.y, self.s, self.z_1, self.z_2, self.t_1, self.t_2

    @staticmethod
    def _ft(t):
        return 2.0 * ((t - 5) ** 4 / 600 + np.exp(-((t - 5) / 0.5) ** 2) + t / 10. - 2)
