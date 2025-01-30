
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from core.trainer import DeepPLIV
from sklearn.preprocessing import StandardScaler
from data_creation import SimDataCreatorHighDimension
from utils.helpers import plot_boxplot


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
                                          dropout: float = 0.0, interaction=False):
    """
    Run the genetic simulation for the high-dimensional case.
     The simulation generates data using the SimDataCreatorHighDimension class
     and estimates the Naive SR-IV, SPS, and SRI models.
    :param num_simulations: parameter to control the number of simulations.
    :param beta_u: parameter to control the coefficient for the confounding variable.
    the endogenous and exogenous variables.
    :param beta_2: parameter to control the coefficient for the exogenous variable.
    :param beta_1: parameter to control the coefficient for the endogenous variable.
    :param n: parameter to control the number of samples.
    :param gamma_u: parameter to control the coefficient for the confounding variable.
    :param learning_rate: parameter to control the learning rate for the neural network model.
    :param epochs: parameter to control the number of epochs for training the neural network model.
    :param k: parameter to control the number of trainings for the neural network model.
    :param p_thr: parameter to control the threshold for the p-value.
    :param m: parameter to control the number of instrumental variables.
    :param dropout: parameter to control the dropout rate for the neural network model.
    :param interaction: parameter to control the interaction term between the endogenous and exogenous variables.
    :return: pd.DataFrame, the results of the simulation.
    """
    results = {
        'method': [],
        'coefficient': [],
        'value': []
    }

    for _ in range(num_simulations):
        betas = (beta_1, beta_2, beta_u)
        data_creator = SimDataCreatorHighDimension(n=n, m = m, betas=betas, gamma_u=gamma_u, p_thr=p_thr,
                                                   interaction=interaction)

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
    results_df.to_pickle(f'high_dim_sim_linear/{n}_{num_simulations}_{beta_1}_{beta_u}_{interaction}.pkl')
    return results_df



if __name__ == '__main__':
    _num_simulations: int = 100
    # This is the simulation for the first only the first part being non-linear
    _beta_1: float = 1
    _k: int = 1
    _lr: float = 0.001
    _gamma_u = 1
    _m = 200000
    _p_thr = 0.05e-6
    _beta_u = 1
    _interaction = False
    for _n in [2000, 10000, 20000, 40000]:
        _dropout: float =  0.3
        _epochs: int = int((1.5 * 10 ** 7) / (_n//2))
        print(f'n: {_n}, dropout: {_dropout}, beta_u: {_beta_u},')

        res = run_high_dimension_genetic_simulation(num_simulations=_num_simulations,
                                                    gamma_u=_gamma_u,
                                                    beta_u=_beta_u,
                                                    n=_n,
                                                    m = _m,
                                                    p_thr=_p_thr,
                                                    beta_1=_beta_1,
                                                    learning_rate=_lr,
                                                    epochs = _epochs,
                                                    k = _k,
                                                    dropout=_dropout,
                                                    interaction=_interaction)

    _interaction = True
    for _n in [2000, 10000, 20000, 40000]:
        for _beta_u in [0.5, 2, 8]:
            _dropout: float = 0.5
            _epochs: int = int((1.5 * 10 ** 7) / (_n//2))
            print(f'n: {_n}, dropout: {_dropout}, beta_u: {_beta_u},')

            res = run_high_dimension_genetic_simulation(num_simulations=_num_simulations,
                                                        gamma_u=_gamma_u,
                                                        beta_u=_beta_u,
                                                        n=_n,
                                                        m = _m,
                                                        p_thr=_p_thr,
                                                        beta_1=_beta_1,
                                                        learning_rate=_lr,
                                                        epochs = _epochs,
                                                        k = _k,
                                                        dropout=_dropout,
                                                        interaction=_interaction)






