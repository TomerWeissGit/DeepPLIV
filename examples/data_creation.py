import itertools

import numpy as np
from sklearn.preprocessing import StandardScaler

from scipy.stats import norm
import random

class SimDataCreatorHighDimension:
    """
    this class replicates the logic of 'generate_data_from_regression' in an oop style,
    similar to 'sim_data_creator_high_dimension'.

    attributes
    ----------
    n_x : int
        sample size for the first stage (to estimate gamma_iv).
    n_y : int
        sample size for the second stage (to generate final outcome y).
    m : int
        number of snps to start with.
    betas : tuple
        (beta0, beta1, beta2, beta_u) for the y model.
    gamma_u : tuple
        gamma_u for the x model.
    p_thr : float
        p-value threshold for snp filtering.

    after instantiation, call get_data() to retrieve data arrays:
        x_1, x_2, y, z_1, z_2
    """

    def __init__(self, n, m, betas, gamma_u, p_thr, interaction = False):
        self.n_x = int(n//2)
        self.n_y = int(n//2)
        self.m = int(m)
        self.betas = betas  # (beta0, beta1, beta2, beta_u)
        self.gamma_u = gamma_u  # (gamma_0, gamma_u)
        self.p_thr = p_thr

        # generate all data upon instantiation
        self.x_1, self.x_2, self.z_1, self.z_2, self.g, self.y = self._generate_data(interaction)

    def _generate_data(self, interaction):
        """
        main data-generation pipeline. loosely follows the structure of
        'generate_data_from_regression', divided into:
          1) first stage (n_x) for x
          2) second stage (n_y) for y
        """

        # unpack parameters
        beta1, beta2, beta_u = self.betas
        gamma_u = self.gamma_u

        # first-stage generation (n_x)
        # create m snp effects from n(0, sqrt(5e-3)) distribution
        sigma_iv = np.sqrt(5e-3)
        gamma_ivs = np.random.normal(0, sigma_iv, self.m)

        # create "gwas" estimates for each snp ~ normal(gamma_iv, sqrt(1/n_x))
        gamma_ivs_for_gwas = np.random.normal(loc=gamma_ivs, scale=np.sqrt(1 / self.n_x), size=self.m)

        # filter snps by p-value < p_thr
        p_values = 2 * (1 - norm.cdf(np.sqrt(self.n_x) * np.abs(gamma_ivs_for_gwas)))
        mask = (p_values < self.p_thr)
        gamma_ivs_filtered = gamma_ivs[mask][:500] if len(gamma_ivs[mask]) > 500 else gamma_ivs[mask]
        number_of_ivs = len(gamma_ivs_filtered)

        # generate x for the n_x individuals using filtered snps
        g_iv_n_x = np.random.binomial(n=2, p=0.3, size=(self.n_x, number_of_ivs))
        g_iv_n_y = np.random.binomial(n=2, p=0.3, size=(self.n_y, number_of_ivs))

        # unobserved confounder
        u_n_x = np.random.normal(0, 1, self.n_x)
        u_n_y = np.random.normal(0, 1, self.n_y)

        # x ~  sum_j(gamma_ivs_filtered_j * g_iv_n_x_j) + gamma_u*u + normal(0,1)
        if interaction:
            n_pairs = g_iv_n_x.shape[1]//4
            # number of pairs you want to use
            # (here we just do k//2, but you could do something else)
            # all possible pairs of indices (i < j)
            all_pairs = list(itertools.combinations(range(n_pairs), 2))
            # randomly select n_pairs of them
            chosen_pairs = random.sample(all_pairs, n_pairs)

            x_n_x = self.generate_x_with_interactions(g_iv=g_iv_n_x, gamma_ivs_filtered=gamma_ivs_filtered,
                                                      chosen_pairs=chosen_pairs) + gamma_u * u_n_x
            x_n_y = self.generate_x_with_interactions(g_iv=g_iv_n_y, gamma_ivs_filtered=gamma_ivs_filtered,
                                                      chosen_pairs=chosen_pairs) + gamma_u * u_n_y
        else:
            x_n_x = g_iv_n_x @ gamma_ivs_filtered + gamma_u * u_n_x
            x_n_y = g_iv_n_y @ gamma_ivs_filtered + gamma_u * u_n_y
        # 2) second-stage generation (n_y)
        # create a single snp g that interacts with x
        g = np.random.binomial(2, 0.25, self.n_y)
        # mean-center g
        g = g - np.mean(g)
        if interaction:
            scaler = StandardScaler()
            y = (beta1 * x_n_y + (np.exp(-beta2 * g + 5)) ** 1.2 * ((g ** 2) ** 0.5) + beta_u * u_n_y
                 + np.random.normal(0, 1, self.n_y))
        else:
            y = beta1 * x_n_y + beta2 * g  + beta_u * u_n_y + np.random.normal(0, 1, self.n_y)

        # for demonstration, let z_2 be the set of original gamma_ivs (pre-filter)
        return x_n_x, x_n_y, g_iv_n_x, g_iv_n_y, g, y

    def get_data(self) -> tuple:
        """
        return data in a structure similar to the original class:
            x_1, x_2, z_1, z_2, g, y
        """
        return (
            self.x_1,
            self.x_2,
            self.z_1,
            self.z_2,
            self.g,
            self.y
        )
    @staticmethod
    def generate_x_with_interactions(
            g_iv: np.ndarray,
            gamma_ivs_filtered: np.ndarray,
            chosen_pairs: np.array,
    ) -> np.ndarray:
        """
        generate x from interactions among random pairs of ivs,
        plus a confounder gamma_u * u_n_x, plus normal(0, 1) noise.

        parameters
        ----------
        g_iv : np.ndarray
            (n_x, k) genotype matrix after mean-centering,
            where k = len(gamma_ivs_filtered).
        gamma_ivs_filtered : np.ndarray
            the array of iv effects (gamma_i), length = k.
        chosen_pairs : np.array
        returns
        -------
        x : np.ndarray
            shape (n_x,). x generated from pairwise snp interactions,
            confounder, and normal(0,1) noise.
        """

        # initialize x contribution from interactions
        x_contrib = np.zeros(g_iv.shape[0])

        # accumulate pairwise interaction effects
        for (i, j) in chosen_pairs:
            interaction_effect = ((np.abs(gamma_ivs_filtered[i]) + np.abs(gamma_ivs_filtered[j]))
                                  * (np.sign(gamma_ivs_filtered[i])) * np.sign(gamma_ivs_filtered[j]))
            x_contrib += interaction_effect * (g_iv[:, i] * g_iv[:, j])

        # add confounder and random noise
        x = x_contrib

        return x


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
