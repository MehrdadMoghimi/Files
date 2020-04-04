from scipy.ndimage.measurements import label
import numpy as np
from scipy.stats import norm
from arch import arch_model
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import seed
import sys


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
    PM = ((1 - alpha / 100.0) * violation_space_PM(returns, VARs) + (alpha / 100.0) * safe_space_PM(returns, VARs)) / np.sum(returns < 0)
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


def forecast_std(portfolio_returns, window_length, volatility_model='GARCH', dist='Normal'):
    # volatility_model = {GARCH, EGARCH, ARCH, HARCH, GJR-GARCH, TARCH}
    # dist = {Normal, t, skewt, ged}
    portfolio_returns = portfolio_returns * 100.0
    test_len = portfolio_returns.shape[0] - window_length
    forecast_std_arch = np.zeros(test_len)
    forecast_mean_arch = np.zeros(test_len)
    print('\n' + volatility_model + ':')
    if volatility_model == 'GJR-GARCH':
        for j in range(0, test_len):
            progressBar(j, test_len, bar_length=20)
            window = portfolio_returns[j:j + window_length]
            arch = arch_model(window, p=1, o=1, q=1)
            arch_fit = arch.fit(disp="off")
            arch_forecast = arch_fit.forecast(horizon=1)
            forecast_mean_arch[j] = arch_forecast.mean.iloc[-1, 0]
            forecast_std_arch[j] = np.sqrt(arch_forecast.variance.iloc[-1, 0])
    elif volatility_model == 'TARCH':
        for j in range(0, test_len):
            progressBar(j, test_len, bar_length=20)
            window = portfolio_returns[j:j + window_length]
            arch = arch_model(window, p=1, o=1, q=1, power=1.0)
            arch_fit = arch.fit(disp="off")
            arch_forecast = arch_fit.forecast(horizon=1)
            forecast_mean_arch[j] = arch_forecast.mean.iloc[-1, 0]
            forecast_std_arch[j] = np.sqrt(arch_forecast.variance.iloc[-1, 0])
    else:
        for j in range(0, test_len):
            progressBar(j, test_len, bar_length=20)
            window = portfolio_returns[j:j + window_length]
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


def calculate_Var_Covar_VAR(portfolio_returns, window_length, alpha):
    test_len = portfolio_returns.shape[0] - window_length
    forecast_std_n = np.zeros(test_len)
    print('\nVar Covar:')
    for j in range(0, test_len):
        progressBar(j, test_len, bar_length=20)
        window = portfolio_returns[j:j + window_length]
        forecast_std_n[j] = np.std(window)
    std_VaR = norm.ppf(alpha / 100.0) * forecast_std_n
    return std_VaR


def calculate_RiskMetrics_VAR(portfolio_returns, window_length, alpha):
    forecast_std_risk_metric = np.zeros(portfolio_returns.shape[0])
    forecast_std_risk_metric[0] = portfolio_returns[0]
    lambda_risk_metric = 0.94
    print('\nRisk Metrics:')
    for i in range(1, portfolio_returns.shape[0]):
        progressBar(i, portfolio_returns.shape[0], bar_length=20)
        forecast_std_risk_metric[i] = np.sqrt(
            (1 - lambda_risk_metric) * portfolio_returns[i] ** 2 + lambda_risk_metric * forecast_std_risk_metric[
                i - 1] ** 2)
    forecast_std_test = forecast_std_risk_metric[window_length:]
    risk_metric_VaR = norm.ppf(alpha / 100.0) * forecast_std_test
    return risk_metric_VaR


def calculate_GARCH_VAR(portfolio_returns, window_length, alpha):
    forecast_mean_garch, forecast_std_garch = forecast_std(portfolio_returns, window_length, volatility_model='GARCH')
    garch_VaR = forecast_mean_garch + norm.ppf(alpha / 100.0) * forecast_std_garch
    return garch_VaR


def calculate_FIGARCH_VAR(portfolio_returns, window_length, alpha):
    forecast_mean_figarch, forecast_std_figarch = forecast_std(portfolio_returns, window_length,
                                                               volatility_model='FIGARCH')
    garch_VaR = forecast_mean_figarch + norm.ppf(alpha / 100.0) * forecast_std_figarch
    return garch_VaR


def calculate_EGARCH_VAR(portfolio_returns, window_length, alpha):
    forecast_mean_egarch, forecast_std_egarch = forecast_std(portfolio_returns, window_length,
                                                             volatility_model='EGARCH')
    egarch_VaR = forecast_mean_egarch + norm.ppf(alpha / 100.0) * forecast_std_egarch
    return egarch_VaR


def calculate_ARCH_VAR(portfolio_returns, window_length, alpha):
    forecast_mean_arch, forecast_std_arch = forecast_std(portfolio_returns, window_length, volatility_model='ARCH')
    fiarch_VaR = forecast_mean_arch + norm.ppf(alpha / 100.0) * forecast_std_arch
    return fiarch_VaR


def calculate_HARCH_VAR(portfolio_returns, window_length, alpha):
    forecast_mean_harch, forecast_std_harch = forecast_std(portfolio_returns, window_length, volatility_model='HARCH')
    harch_VaR = forecast_mean_harch + norm.ppf(alpha / 100.0) * forecast_std_harch
    return harch_VaR


def calculate_TARCH_VAR(portfolio_returns, window_length, alpha):
    forecast_mean_tarch, forecast_std_tarch = forecast_std(portfolio_returns, window_length, volatility_model='TARCH')
    harch_VaR = forecast_mean_tarch + norm.ppf(alpha / 100.0) * forecast_std_tarch
    return harch_VaR


def calculate_GJR_GARCH_VAR(portfolio_returns, window_length, alpha):
    forecast_mean_tarch, forecast_std_tarch = forecast_std(portfolio_returns, window_length,
                                                           volatility_model='GJR-GARCH')
    harch_VaR = forecast_mean_tarch + norm.ppf(alpha / 100.0) * forecast_std_tarch
    return harch_VaR


def calculate_Historical_VAR(portfolio_returns, window_length, alpha):
    test_len = portfolio_returns.shape[0] - window_length
    hist_VaR = np.zeros((test_len, 1))
    print('\nHistorical:')
    for j in range(test_len):
        progressBar(j, test_len, bar_length=20)
        window = portfolio_returns[j:j + window_length]
        hist_VaR[j] = np.percentile(window, alpha)
    return hist_VaR


def calculate_Filtered_Historical_VAR(portfolio_returns, window_length, alpha, forecast_mean_arima, forecast_std_garch):
    test_len = portfolio_returns.shape[0] - window_length
    f_hist_VaR = np.zeros((test_len, 1))
    print('\nFiltered Historical:')
    for j in range(test_len):
        progressBar(j, test_len, bar_length=20)
        window = portfolio_returns[j:j + window_length]
        MEAN = forecast_mean_arima[j]
        STD = forecast_std_garch[j]
        filtered_window = MEAN + (window - np.mean(window)) * (STD/np.std(window))
        f_hist_VaR[j] = np.percentile(filtered_window, alpha)
    return f_hist_VaR


def calculate_MonteCarlo_VAR(alpha, forecast_mean_arima, forecast_std_garch):
    seed(0)
    n_samples = 10000
    STD = forecast_std_garch.reshape((-1, 1))
    MEAN = forecast_mean_arima.reshape((-1, 1))
    raw_samples = np.random.normal(0, 1, size=(1, n_samples))
    samples_with_scale = np.repeat(MEAN, n_samples, axis=1) + np.dot(STD, raw_samples)
    mc_VaR = np.percentile(samples_with_scale, q=alpha, axis=1).reshape(-1, 1)
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


def calculate_var_models(portfolio_returns, window_length, alpha, forecast_mean_arima, forecast_std_garch):
    var_models = pd.DataFrame()
    # var_models['CAViaR_Sym'] = calculate_CAViaR_Sym_VAR(portfolio_returns, window_length, alpha)
    # var_models['CAViaR_Asym'] = calculate_CAViaR_Asym_VAR(portfolio_returns, window_length, alpha)
    # var_models['CAViaR_indirect_GARCH'] = calculate_CAViaR_indirect_GARCH_VAR(portfolio_returns, window_length, alpha)
    # var_models['CAViaR_adaptive'] = calculate_CAViaR_adaptive_VAR(portfolio_returns, window_length, alpha)
    # var_models['Var_Covar'] = calculate_Var_Covar_VAR(portfolio_returns, window_length, alpha)
    var_models['RiskMetrics'] = calculate_RiskMetrics_VAR(portfolio_returns, window_length, alpha)
    var_models['Historical'] = calculate_Historical_VAR(portfolio_returns, window_length, alpha)
    var_models['F_Historical'] = calculate_Filtered_Historical_VAR(portfolio_returns, window_length, alpha, forecast_mean_arima, forecast_std_garch)
    var_models['MonteCarlo'] = calculate_MonteCarlo_VAR(alpha, forecast_mean_arima, forecast_std_garch)
    var_models['GARCH'] = calculate_GARCH_VAR(portfolio_returns, window_length, alpha)
    # var_models['FIGARCH'] = calculate_FIGARCH_VAR(portfolio_returns, window_length, alpha)
    var_models['EGARCH'] = calculate_EGARCH_VAR(portfolio_returns, window_length, alpha)
    # var_models['ARCH'] = calculate_ARCH_VAR(portfolio_returns, window_length, alpha)
    # var_models['HARCH'] = calculate_HARCH_VAR(portfolio_returns, window_length, alpha)
    # var_models['TARCH'] = calculate_TARCH_VAR(portfolio_returns, window_length, alpha)
    var_models['GJR_GARCH'] = calculate_GJR_GARCH_VAR(portfolio_returns, window_length, alpha)
    # var_models['EVT'] = calculate_EVT_VAR(portfolio_returns, window_length, alpha)
    return var_models


def calculate_var_models_pm(portfolio_returns, window_length, var_models, alpha):
    test_returns = portfolio_returns[window_length:]
    var_models_pm = pd.DataFrame(columns=['name', 'PM', 'ratio'])
    for column in var_models.columns:
        var_models_pm = var_models_pm.append({'name': column, 'PM': penalization_measure(test_returns, var_models[column], alpha)}, ignore_index=True)
    var_models_pm['ratio'] = var_models_pm['PM'] / sum(var_models_pm['PM'])
    return var_models_pm


def plot_all(portfolio_returns, window_length, var_models):
    test_returns = portfolio_returns[window_length:]
    for column in var_models.columns:
        plot(test_returns, var_models[column].values, file_name=column)


def predictive_ability_test(test_returns, var_models, benchmark='GARCH'):
    var_models_error = var_models.subtract(test_returns, axis=0)
    # loss is a function of errors, it can be abs or power of 2
    var_models_loss = np.sqrt(np.power(var_models_error, 2))
    # benchmark_loss = var_models_loss[benchmark]
    # var_models_loss_wo_benchmark = var_models_loss.drop(columns=benchmark)
    # kappa_1 = var_models_loss_wo_benchmark.subtract(benchmark_loss, axis=0)
    kappa = var_models_loss.div(np.sum(var_models_loss, axis=1), axis=0)
    H_0 = np.mean(kappa, axis=0)
    W = np.sum(kappa > (1 / var_models_loss.shape[1]))
    p = 0.5
    T = var_models.shape[0]
    W_hat = (W - p * T) / np.sqrt(p * (1 - p) * T)
    return 0


def plot(returns, VARs, file_name=None):
    # Re-add the time series index
    r = pd.Series(returns.squeeze())
    q = pd.Series(VARs.squeeze())

    sns.set_context("paper")
    sns.set_style("whitegrid", {"font.family": "serif", "font.serif": "Computer Modern Roman", "text.usetex": True})

    ax = plt.gca()
    # Hits
    if len(r[r <= q]) > 0:
        r[r <= q].plot(ax=ax, color="red", marker="o", ls="None", figsize=(12, 7))
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
    plt.ylabel("Return")
    # ax.yaxis.grid()
    plt.title(file_name)

    sns.despine()
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

# import os
# directory = os.getcwd()+'/data' #"C:\\Users\\ehsan\Desktop\Torronto Stocks Data\\"
# rets = pd.read_csv(directory + "/returns_TSX.csv", header=0)
# rets['date'] = pd.to_datetime(rets['date'])
# rets = rets.set_index('date')
# n, m = rets.shape
# weights = np.ones((m, 1)) / m
# portfolio_returns = np.dot(rets, weights).squeeze()
# window_length = np.floor(0.875 * n).astype('int64')
# alpha = 5
# var_models, var_models_pm = calculate_ratios(portfolio_returns, window_length, alpha)

# evar = pd.read_csv('evar.csv')
# evar['date'] = pd.to_datetime(evar['date'])
# evar = evar.set_index('date')
# var_models['evar'] = pd.Series(evar['0'])

# plot(test_returns, var_models['evar'].values, file_name='1.eVaR')
