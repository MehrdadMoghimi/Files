from scipy.ndimage.measurements import label
import numpy as np
from scipy.stats import norm
from arch import arch_model
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import time


def safe_space_PM(returns, VARs):
    PM = sum((returns < 0) * (returns > VARs) * (returns - VARs))
    return PM


def violation_space_PM(returns, VARs):
    e = VARs - returns
    violations = e > 0
    labels = label(violations)
    num_of_clusters = labels[1]
    clusters = [e[labels[0] == i] for i in range(1, num_of_clusters + 1)]
    cluster_centers = [np.mean(np.where(labels[0] == i)[0]) for i in range(1, num_of_clusters + 1)]
    cluster_quantity = [np.prod(1 + c) for c in clusters]
    PM = 0
    for i in range(num_of_clusters):
        for j in range(i + 1, num_of_clusters):
            PM = PM + (cluster_quantity[i] * cluster_quantity[j] - 1) / (cluster_centers[j] - cluster_centers[i])
    return PM


def penalization_measure(returns, VARs, alpha):
    PM = ((1 - alpha / 100.0) * violation_space_PM(returns, VARs) + (alpha / 100.0) * safe_space_PM(returns,
                                                                                                    VARs)) / np.sum(
        returns < 0)
    return PM


def HELP(returns):
    arch = arch_model(returns, vol='Garch', p=1, o=0, q=1, dist='skewt')
    arch_fit = arch.fit(disp='off', last_obs='2017-12-31')
    arch_forecast = arch_fit.forecast(start='2018-1-1')
    cond_mean = arch_forecast.mean['2018':]
    cond_var = arch_forecast.variance['2018':]
    q = arch.distribution.ppf([0.01, 0.05], parameters=arch_fit.params[-2:])
    print(q)
    value_at_risk = -cond_mean.values - np.sqrt(cond_var).values * q[None, :]
    value_at_risk = pd.DataFrame(value_at_risk, columns=['1%', '5%'], index=cond_var.index)
    return value_at_risk


def forecast_std(returns, weights, window_length, test_len, volatility_model='GARCH', dist='Normal'):
    # volatility_model = {GARCH, EGARCH, GJR-GARCH}
    # dist = {Normal, t, skewt, ged}
    portfolio_returns = returns.dot(weights)
    portfolio_returns = portfolio_returns * 100.0
    forecast_std_arch = np.zeros(test_len)
    forecast_mean_arch = np.zeros(test_len)
    print('\n' + volatility_model + ':')
    if volatility_model == 'GJR-GARCH':
        for j in range(0, test_len):
            loc = returns.shape[0] - test_len + j
            progressBar(j, test_len, bar_length=20)
            window = portfolio_returns.iloc[loc - window_length:loc]
            arch = arch_model(window, p=1, o=1, q=1)
            arch_fit = arch.fit(disp="off")
            arch_forecast = arch_fit.forecast(horizon=1)
            forecast_mean_arch[j] = arch_forecast.mean.iloc[-1, 0]
            forecast_std_arch[j] = np.sqrt(arch_forecast.variance.iloc[-1, 0])
    else:
        for j in range(0, test_len):
            loc = returns.shape[0] - test_len + j
            progressBar(j, test_len, bar_length=20)
            window = portfolio_returns.iloc[loc - window_length:loc]
            arch = arch_model(window, mean='AR', dist=dist, vol=volatility_model, p=1, q=1)
            arch_fit = arch.fit(disp="off")
            arch_forecast = arch_fit.forecast(horizon=1)
            forecast_mean_arch[j] = arch_forecast.mean.iloc[-1, 0]
            forecast_std_arch[j] = np.sqrt(arch_forecast.variance.iloc[-1, 0])
    forecast_mean_arch = forecast_mean_arch / 100.0
    forecast_std_arch = forecast_std_arch / 100.0
    return forecast_mean_arch, forecast_std_arch


def forecast_mean(portfolio_returns, window_length):
    portfolio_returns = portfolio_returns * 100.0
    test_len = portfolio_returns.shape[0] - window_length
    forecast = np.zeros(test_len)
    print('\nForecast mean ARIMA:')
    for j in range(0, test_len):
        progressBar(j, test_len, bar_length=20)
        window = portfolio_returns[j:j + window_length]
        arima = ARIMA(window, order=(1, 1, 1))
        arima_fit = arima.fit(disp=0)
        arima_output = arima_fit.forecast()
        forecast[j] = arima_output[0]
    forecast = forecast / 100.0
    return forecast


def calculate_Var_Covar_VAR(returns, weights, window_length, test_len, alpha):
    t = time.time()
    std_VaR = np.zeros(test_len)
    print('\nVar Covar:')
    for j in range(0, test_len):
        loc = returns.shape[0] - test_len + j
        progressBar(j, test_len, bar_length=20)
        window = returns.iloc[loc - window_length:loc, :]
        cov_matrix = window.cov()
        avg_rets = window.mean()
        port_mean = avg_rets.dot(weights)
        port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
        std_VaR[j] = norm.ppf(alpha / 100.0, port_mean, port_stdev)
    print('\nVar_Covar time: {}'.format(round(time.time() - t, 2)))
    return std_VaR


def calculate_RiskMetrics_VAR(returns, weights, test_len, alpha):
    t = time.time()
    lambda_risk_metric = 0.94
    portfolio_returns = returns.dot(weights)
    forecast_std_risk_metric = pd.Series(portfolio_returns.squeeze()).ewm(alpha=1 - lambda_risk_metric).std().shift(periods=1)
    forecast_std_test = forecast_std_risk_metric[-test_len:]
    risk_metric_VaR = norm.ppf(alpha / 100.0) * forecast_std_test
    print('\nRiskMetrics time: {}'.format(round(time.time() - t, 2)))
    return risk_metric_VaR


def calculate_GARCH_VAR(returns, weights, window_length, test_len, alpha):
    t = time.time()
    forecast_mean_garch, forecast_std_garch = forecast_std(returns, weights, window_length, test_len, volatility_model='GARCH')
    garch_VaR = forecast_mean_garch + norm.ppf(alpha / 100.0) * forecast_std_garch
    print('\nGARCH time: {}'.format(round(time.time() - t, 2)))
    return garch_VaR


def calculate_EGARCH_VAR(returns, weights, window_length, test_len, alpha):
    t = time.time()
    forecast_mean_egarch, forecast_std_egarch = forecast_std(returns, weights, window_length, test_len, volatility_model='EGARCH')
    egarch_VaR = forecast_mean_egarch + norm.ppf(alpha / 100.0) * forecast_std_egarch
    print('\nEGARCH time: {}'.format(round(time.time() - t, 2)))
    return egarch_VaR


def calculate_GJR_GARCH_VAR(returns, weights, window_length, test_len, alpha):
    t = time.time()
    forecast_mean_tarch, forecast_std_tarch = forecast_std(returns, weights, window_length, test_len, volatility_model='GJR-GARCH')
    harch_VaR = forecast_mean_tarch + norm.ppf(alpha / 100.0) * forecast_std_tarch
    print('\nGJR GARCH time: {}'.format(round(time.time() - t, 2)))
    return harch_VaR


def calculate_Historical_VAR(returns, weights, window_length, test_len, alpha):
    t = time.time()
    hist_VaR = np.zeros((test_len, 1))
    portfolio_returns = returns.dot(weights)
    print('\nHistorical:')
    for j in range(test_len):
        progressBar(j, test_len, bar_length=20)
        window = portfolio_returns[j:j + window_length]
        hist_VaR[j] = np.percentile(window, alpha)
    print('\nHistorical time: {}'.format(round(time.time() - t, 2)))
    return hist_VaR


def calculate_Filtered_Historical_VAR(returns, weights, window_length, test_len, alpha, forecast_mean, forecast_std):
    t = time.time()
    f_hist_VaR = np.zeros((test_len, 1))
    portfolio_returns = returns.dot(weights)
    print('\nFiltered Historical:')
    for j in range(test_len):
        loc = returns.shape[0] - test_len + j
        progressBar(j, test_len, bar_length=20)
        window = portfolio_returns.iloc[loc - window_length:loc]
        MEAN = forecast_mean[j]
        STD = forecast_std[j]
        filtered_window = MEAN + (window - np.mean(window)) * (STD / np.std(window))
        f_hist_VaR[j] = np.percentile(filtered_window, alpha)
    print('\nFiltered Historical time: {}'.format(round(time.time() - t, 2)))
    return f_hist_VaR


def calculate_MonteCarlo_VAR(returns, weights, window_length, test_len, alpha):
    t = time.time()
    n_samples = 10000
    mc_VaR = np.zeros(test_len)
    print('\nMonte Carlo:')
    for j in range(0, test_len):
        loc = returns.shape[0] - test_len + j
        progressBar(j, test_len, bar_length=20)
        window = returns.iloc[loc - window_length:loc, :]
        log_return = np.random.multivariate_normal(window.mean() - (window.std() ** 2) / 2, window.cov(), size=n_samples)
        port_simulations = (np.exp(log_return) - 1).dot(weights)
        mc_VaR[j] = np.percentile(port_simulations, alpha)
    print('\nMonte Carlo time: {}'.format(round(time.time() - t, 2)))
    return mc_VaR


def calculate_CAViaR_Sym_VAR():
    return 0


def calculate_CAViaR_Asym_VAR():
    return 0


def calculate_CAViaR_indirect_GARCH_VAR():
    return 0


def calculate_CAViaR_adaptive_VAR():
    return 0


def calculate_EVT_VAR():
    return 0


def calculate_var_models(returns, weights, window_length, test_len, alpha, forecast_mean, forecast_std):
    var_models = pd.DataFrame()
    # var_models['CAViaR_Sym'] = calculate_CAViaR_Sym_VAR(portfolio_returns, window_length, alpha)
    # var_models['CAViaR_Asym'] = calculate_CAViaR_Asym_VAR(portfolio_returns, window_length, alpha)
    # var_models['CAViaR_indirect_GARCH'] = calculate_CAViaR_indirect_GARCH_VAR(portfolio_returns, window_length, alpha)
    # var_models['CAViaR_adaptive'] = calculate_CAViaR_adaptive_VAR(portfolio_returns, window_length, alpha)
    var_models['RiskMetrics'] = calculate_RiskMetrics_VAR(returns, weights, test_len, alpha)
    var_models['Variance - Covariance'] = calculate_Var_Covar_VAR(returns, weights, round(window_length/10), test_len, alpha)
    var_models['Historical'] = calculate_Historical_VAR(returns, weights, round(window_length/4), test_len, alpha)
    var_models['Filtered Historical'] = calculate_Filtered_Historical_VAR(returns, weights, window_length, test_len, alpha, forecast_mean, forecast_std)
    var_models['Monte Carlo'] = calculate_MonteCarlo_VAR(returns, weights, round(window_length/10), test_len, alpha)
    var_models['GARCH'] = calculate_GARCH_VAR(returns, weights, window_length, test_len, alpha)
    var_models['E-GARCH'] = calculate_EGARCH_VAR(returns, weights, window_length, test_len, alpha)
    # var_models['GJR_GARCH'] = calculate_GJR_GARCH_VAR(portfolio_returns, window_length, alpha)
    # var_models['EVT'] = calculate_EVT_VAR(portfolio_returns, window_length, alpha)
    return var_models


def calculate_var_models_pm(test_returns, var_models, alpha):
    var_models_pm = pd.DataFrame(columns=['name', 'PM', 'ratio'])
    for column in var_models.columns:
        var_models_pm = var_models_pm.append(
            {'name': column, 'PM': penalization_measure(test_returns, var_models[column], alpha)}, ignore_index=True)
    var_models_pm['ratio'] = var_models_pm['PM'] / sum(var_models_pm['PM'])
    var_models_pm.set_index('name', inplace=True, drop=True)
    return var_models_pm


def plot_all(test_returns, var_models):
    for column in var_models.columns:
        plot(test_returns.values, var_models[column].values, file_name=column)


def predictive_ability_test(test_returns, var_models, alpha, loss_func):
    var_models_error = var_models.subtract(test_returns, axis=0)
    # loss is a function of errors, it can be abs or power of 2
    if loss_func == 'mse':
        var_models_loss = np.sqrt(np.power(var_models_error, 2))
    elif loss_func == 'abs':
        var_models_loss = np.abs(var_models_error)
    elif loss_func == 'regulatory':
        var_models_loss = ((np.repeat(test_returns.values.reshape(-1, 1), var_models.shape[1], axis=1) < var_models) * 1 - alpha / 100) * var_models_error
    elif loss_func == 'quantile':
        var_models_loss = pd.DataFrame(index=var_models.index, columns=var_models.columns)
        for column in var_models.columns:
            var_model = var_models[column]
            QL = []
            for i in range(len(var_model) - 1):
                if test_returns[i] < var_model[i]:
                    QuantileLoss = (test_returns[i] - var_model[i]) ** 2
                else:
                    QuantileLoss = (test_returns[i + 1:].quantile(alpha / 100) - var_model[i]) ** 2
                QL.append(QuantileLoss)
            QL.append((test_returns[-1] - var_model[-1]) ** 2)
            var_models_loss[column] = QL
    else:
        return "loss function must be one of quantile, mse, abs or regulatory"
    kappa = var_models_loss.div(np.sum(var_models_loss, axis=1), axis=0)
    W = np.sum(kappa > (1 / var_models_loss.shape[1]))
    p = 0.5
    T = var_models.shape[0]
    W_hat = (W - p * T) / np.sqrt(p * (1 - p) * T)
    PAT = W_hat.to_frame(name='W_hat')
    PAT['p-value'] = [norm.cdf(x) for x in PAT['W_hat']]
    return PAT


def plot(returns, VARs, file_name=None):
    # Re-add the time series index
    r = pd.Series(returns.squeeze())
    q = pd.Series(VARs.squeeze())

    sns.set_context("paper")
    sns.set_style("whitegrid", {"font.family": "serif", "font.serif": "Computer Modern Roman", "text.usetex": True})

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.spines["top"].set_linewidth(2)
    ax.spines["top"].set_color("black")
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(2)
    ax.spines["left"].set_color("black")
    ax.spines["right"].set_linewidth(2)
    ax.spines["right"].set_color("black")
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    # Hits
    if len(r[r <= q]) > 0:
        r[r <= q].plot(ax=ax, color="red", marker="o", ls="None")
        for h in r[r <= q].index:
            plt.axvline(h, color="black", alpha=0.4, linewidth=1, zorder=0)

    # Positive returns
    if len(r[q < r]) > 0:
        r[q < r].plot(ax=ax, color="green", marker="o", ls="None")

    # Negative returns but no hit
    if len(r[(q <= r) & (r <= 0)]) > 0:
        r[(q <= r) & (r <= 0)].plot(ax=ax, color="orange", marker="o", ls="None")

    # VaR
    q.plot(ax=ax, grid=False, color="black", rot=0)

    # Axes
    plt.xlabel("")
    plt.ylabel("")
    # ax.yaxis.grid()
    # plt.title(file_name)

    #sns.despine()
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name + '.png', bbox_inches="tight")
    plt.close("all")


def progressBar(value, end_value, bar_length=20):
    percent = float(value) / end_value
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rCompleted: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()
