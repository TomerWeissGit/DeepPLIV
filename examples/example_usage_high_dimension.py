import itertools
import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from deeppliv.core.trainer import DeepPLIV
from sklearn.preprocessing import StandardScaler
from data_creation import SimDataCreatorHighDimension
from deeppliv.utils.helpers import plot_boxplot

# Module-level globals to persist interaction structure across all simulations
FIXED_CHOSEN_PAIRS = None
FIXED_INTERACTION_EFFECTS = None


def generate_random_pairs(k, n_pairs):
    pairs = []
    while len(pairs) < n_pairs:
        i, j = sorted(np.random.choice(k, 2, replace=False))
        pair = (i, j)
        if pair not in pairs:
            pairs.append(pair)
    return pairs


# Initialize fixed interaction logic once (can also be replaced with file-based cache if needed)
def initialize_fixed_interactions():
    global FIXED_CHOSEN_PAIRS, FIXED_INTERACTION_EFFECTS
    if FIXED_CHOSEN_PAIRS is None or FIXED_INTERACTION_EFFECTS is None:
        print('hellllll00000000')
        dummy_k = 500
        n_pairs = dummy_k // 4
        FIXED_CHOSEN_PAIRS = generate_random_pairs(k=dummy_k, n_pairs=n_pairs)

        dummy_gamma = np.random.normal(0, np.sqrt(5e-3), dummy_k)
        FIXED_INTERACTION_EFFECTS = [
            (np.abs(dummy_gamma[i]) + np.abs(dummy_gamma[j])) *
            np.sign(dummy_gamma[i]) * np.sign(dummy_gamma[j])
            for (i, j) in FIXED_CHOSEN_PAIRS
        ]


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
        self.x_1, self.x_2, self.z_1, self.z_2, self.g, self.y = data_creator.get_data()
        self.dropout = dropout
        self.xa = self.z_1
        self.xb = self.z_2

    def estimate_naive_regression(self):
        """
        Estimate the Naive SR-IV model using a simple linear regression model.
        :return: np.array, the coefficients of the Naive SR-IV model.
        """
        x = np.concatenate((self.x_2.reshape(-1, 1),
                            self.g.reshape(-1, 1),
                            (self.x_2 * self.g).reshape(-1, 1)), axis=1)
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
                            self.g.reshape(-1, 1)),
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
                            self.g.reshape(-1, 1),
                            (self.x_2 * self.g).reshape(-1, 1),
                            x_error.reshape(-1, 1)],
                           axis=1)
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
                                                  validation_data=(g_iv_2, x_2))
        x_predicted = first_stage_model.predict(g_iv_2).reshape(1, -1)
        x_error = self.x_2 - x_predicted
        x_predicted = x_predicted
        scaler = StandardScaler()
        x_exog = scaler.fit_transform(np.concatenate((self.g.reshape(-1, 1),
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
        x_exog = self.g.reshape(-1, 1)

        model_sps = model.fit_second_stage(x_predicted.reshape(-1, 1),
                                           x_exog,
                                           self.y.reshape(-1, 1),
                                           epochs_second_stage=self.epochs,
                                           learning_rate_second_stage=self.learning_rate,
                                           dropout=self.dropout)

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
                                          m: int = 1000,
                                          p_thr: float = 0.05,
                                          beta_1: float = 1,
                                          gamma_u: float = 1,
                                          beta_u: float = 1,
                                          beta_2: float = 1,
                                          n: int = 20000,
                                          learning_rate: float = 0.01,
                                          epochs: int = 2000,
                                          k: int = 5,
                                          dropout: float = 0.0,
                                          interaction=False,
                                          chosen_pairs=None,
                                          interaction_effects=None):
    results = {
        'method': [],
        'coefficient': [],
        'value': []
    }

    for _ in range(num_simulations):
        betas = (beta_1, beta_2, beta_u)
        data_creator = SimDataCreatorHighDimension(n=n,
                                                   m=m,
                                                   betas=betas,
                                                   gamma_u=gamma_u,
                                                   p_thr=p_thr,
                                                   interaction=interaction,
                                                   chosen_pairs=chosen_pairs,
                                                   interaction_effects=interaction_effects)

        naive_srisps = NaiveSRISPSHighDimension(data_creator=data_creator,
                                                epochs=epochs,
                                                learning_rate=learning_rate,
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
                    print(f'coef_{i} {method.lower()}: {coef:.3f}')

        coefficients_sri_lst, coefficients_sps_lst, coefficients_nff_lst = [], [], []

        for i in range(k):
            coefficients_sps, coefficients_sri, coefficients_nff = naive_srisps.estimating_sri_sps_with_nn()
            coefficients_sri_lst.append(coefficients_sri)
            coefficients_sps_lst.append(coefficients_sps)
            coefficients_nff_lst.append(coefficients_nff)

        for method, coefs in zip(['SPS with NN', 'SRI with NN', 'Naive Feed Forward'],
                                 [coefficients_sps, coefficients_sri, coefficients_nff]):
            if np.abs(coefs[0]) < 10:
                results['method'].append(method)
                results['coefficient'].append('coef_0')
                results['value'].append(coefs[0])
                print(f'coef_0 {method.lower()}: {coefs[0]:.3f}')

        for method, coefs in zip([f'SPS with NN - mean {k}',
                                  f'SRI with NN - mean {k}',
                                  f'Naive Feed Forward - mean {k}'],
                                 [np.mean(coefficients_sps_lst, axis=0),
                                  np.mean(coefficients_sri_lst, axis=0),
                                  np.mean(coefficients_nff_lst, axis=0)]):
            if np.abs(coefs[0]) < 10:
                results['method'].append(method)
                results['coefficient'].append('coef_0')
                results['value'].append(coefs[0])

    return pd.DataFrame(results)


from concurrent.futures import ProcessPoolExecutor


def simulate_single_setting(n, beta_u, dropout, epochs, config):
    beta_1, gamma_u, m, p_thr, num_simulations, learning_rate, k, interaction, chosen_pairs, interaction_effects = config

    df = run_high_dimension_genetic_simulation(
        num_simulations=num_simulations,
        gamma_u=gamma_u,
        beta_u=beta_u,
        n=n,
        m=m,
        p_thr=p_thr,
        beta_1=beta_1,
        learning_rate=learning_rate,
        epochs=epochs,
        k=k,
        dropout=dropout,
        interaction=interaction,
        chosen_pairs=chosen_pairs,
        interaction_effects=interaction_effects  # <-- this is crucial
    )

    return (n, beta_u, df)


def run_single_simulation(sim_id, n, beta_u, dropout, epochs, config):
    beta_1, gamma_u, m, p_thr, learning_rate, k, interaction, chosen_pairs, interaction_effects = config

    df = run_high_dimension_genetic_simulation(
        num_simulations=1,  # just 1 per job!
        gamma_u=gamma_u,
        beta_u=beta_u,
        n=n,
        m=m,
        p_thr=p_thr,
        beta_1=beta_1,
        learning_rate=learning_rate,
        epochs=epochs,
        k=k,
        dropout=dropout,
        interaction=interaction,
        chosen_pairs=chosen_pairs,
        interaction_effects=interaction_effects
    )

    return df


def run_parallel_simulations_for_config(n, beta_u, dropout, epochs, config, num_simulations):
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(i, n, beta_u, dropout, epochs, config) for i in range(num_simulations)]

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_simulation, *job) for job in jobs]
        for future in futures:
            df = future.result()
            results.append(df)

    combined_df = pd.concat(results, ignore_index=True)
    return combined_df


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.set_start_method('spawn')  # for safety on MacOS/Windows

    _num_simulations = 8
    _beta_1 = 1
    _k = 1
    _lr = 0.001
    _gamma_u = 1
    _m = 200000
    _p_thr = 0.05e-6
    _interaction = True

    dummy_k = 500
    n_pairs = dummy_k // 10
    fixed_chosen_pairs = generate_random_pairs(dummy_k, n_pairs)
    dummy_gamma = np.random.normal(0, 5e-4, dummy_k)
    fixed_effects = [
        1.0 if (np.abs(dummy_gamma[i]) > 1e-3 and np.abs(dummy_gamma[j]) < 1e-4) else 0.0
        for (i, j) in fixed_chosen_pairs
    ]
    config = (_beta_1, _gamma_u, _m, _p_thr, _lr, _k, _interaction,
              fixed_chosen_pairs, fixed_effects)

    for _n in [10000]:
        for _beta_u in [1]:
            _dropout = 0.5
            _epochs = int((1.5 * 10 ** 7) / (_n // 2))

            print(f"🚀 Running {_num_simulations} parallel simulations for n={_n}, beta_u={_beta_u}")
            df = run_parallel_simulations_for_config(
                n=_n,
                beta_u=_beta_u,
                dropout=_dropout,
                epochs=_epochs,
                config=config,
                num_simulations=_num_simulations
            )
            print(f"✅ Finished: n={_n}, beta_u={_beta_u}")
            plot_boxplot(df)
