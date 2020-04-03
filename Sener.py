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
    PM = sum((returns < 0)*(returns > VARs)*(returns-VARs))
    return PM


def violation_space_PM(returns, VARs):
    e = VARs - returns
    violations = e > 0
    labels = label(violations)
    num_of_clusters = labels[1]
    clusters = [e[labels[0] == i] for i in range(1, num_of_clusters+1)]
    cluster_centers = [np.mean(np.where(labels[0] == i)[0]) for i in range(1, num_of_clusters+1)]
    cluster_quantity = [np.prod(1+c) for c in clusters]
    PM = 0
    for i in range(num_of_clusters):
        for j in range(i+1, num_of_clusters):
            PM = PM + (cluster_quantity[i]*cluster_quantity[j]-1)/(cluster_centers[j]-cluster_centers[i])
    return PM


def penalization_measure(returns, VARs, alpha):
    PM = ((1-alpha/100.0)*violation_space_PM(returns, VARs) + (alpha/100.0)*safe_space_PM(returns, VARs))/np.sum(returns<0)
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
    portfolio_returns = portfolio_returns*1000
    test_len = portfolio_returns.shape[0] - window_length
    forecast_std_arch = np.zeros(test_len)
    forecast_mean_arch = np.zeros(test_len)
    if volatility_model == 'GJR-GARCH':
        for j in range(0, test_len):
            print(volatility_model)
            progressBar(j, test_len, bar_length=20)
            window = portfolio_returns[j:j + window_length]
            arch = arch_model(window, p=1, o=1, q=1)
            arch_fit = arch.fit(disp="off")
            arch_forecast = arch_fit.forecast(horizon=1)
            forecast_mean_arch[j] = arch_forecast.mean.iloc[-1, 0]
            forecast_std_arch[j] = np.sqrt(arch_forecast.variance.iloc[-1, 0])
    elif volatility_model == 'TARCH':
        for j in range(0, test_len):
            print(volatility_model)
            progressBar(j, test_len, bar_length=20)
            window = portfolio_returns[j:j + window_length]
            arch = arch_model(window, p=1, o=1, q=1, power=1.0)
            arch_fit = arch.fit(disp="off")
            arch_forecast = arch_fit.forecast(horizon=1)
            forecast_mean_arch[j] = arch_forecast.mean.iloc[-1, 0]
            forecast_std_arch[j] = np.sqrt(arch_forecast.variance.iloc[-1, 0])
    else:
        for j in range(0, test_len):
            print(volatility_model)
            progressBar(j, test_len, bar_length=20)
            window = portfolio_returns[j:j + window_length]
            arch = arch_model(window, mean='AR', dist=dist, vol=volatility_model, p=1, q=1)
            arch_fit = arch.fit(disp="off")
            arch_forecast = arch_fit.forecast(horizon=1)
            forecast_mean_arch[j] = arch_forecast.mean.iloc[-1, 0]
            forecast_std_arch[j] = np.sqrt(arch_forecast.variance.iloc[-1, 0])
    forecast_mean_arch = forecast_mean_arch/1000.0
    forecast_std_arch = forecast_std_arch/1000.0
    return forecast_mean_arch, forecast_std_arch


def forecast_mean(portfolio_returns, window_length):
    portfolio_returns = portfolio_returns * 1000
    test_len = portfolio_returns.shape[0] - window_length
    forecast = np.zeros(test_len)
    for j in range(0, test_len):
        print('forecast_mean')
        progressBar(j, test_len, bar_length=20)
        window = portfolio_returns[j:j + window_length]
        arima = ARIMA(window, order=(1, 1, 1))
        arima_fit = arima.fit(disp=0)
        arima_output = arima_fit.forecast()
        forecast[j] = arima_output[0]
    forecast = forecast/1000
    return forecast


def calculate_Var_Covar_VAR(portfolio_returns, window_length, alpha):
    test_len = portfolio_returns.shape[0] - window_length
    forecast_std_n = np.zeros(test_len)
    for j in range(0, test_len):
        window = portfolio_returns[j:j + window_length]
        forecast_std_n[j] = np.std(window)
    std_VaR = norm.ppf(alpha / 100.0) * forecast_std_n
    return std_VaR


def calculate_RiskMetrics_VAR(portfolio_returns, window_length, alpha):
    forecast_std_risk_metric = np.zeros(portfolio_returns.shape[0])
    forecast_std_risk_metric[0] = portfolio_returns[0]
    lambda_risk_metric = 0.94
    for i in range(1, portfolio_returns.shape[0]):
        forecast_std_risk_metric[i] = np.sqrt((1-lambda_risk_metric)*portfolio_returns[i]**2 + lambda_risk_metric*forecast_std_risk_metric[i-1]**2)
    forecast_std_test = forecast_std_risk_metric[window_length:]
    risk_metric_VaR = norm.ppf(alpha / 100.0) * forecast_std_test
    return risk_metric_VaR


def calculate_GARCH_VAR(portfolio_returns, window_length, alpha):
    forecast_mean_garch, forecast_std_garch = forecast_std(portfolio_returns, window_length, volatility_model='GARCH')
    garch_VaR = forecast_mean_garch + norm.ppf(alpha/100.0)*forecast_std_garch
    return garch_VaR


def calculate_FIGARCH_VAR(portfolio_returns, window_length, alpha):
    forecast_mean_figarch, forecast_std_figarch = forecast_std(portfolio_returns, window_length, volatility_model='FIGARCH')
    garch_VaR = forecast_mean_figarch + norm.ppf(alpha/100.0)*forecast_std_figarch
    return garch_VaR


def calculate_EGARCH_VAR(portfolio_returns, window_length, alpha):
    forecast_mean_egarch, forecast_std_egarch = forecast_std(portfolio_returns, window_length, volatility_model='EGARCH')
    egarch_VaR = forecast_mean_egarch + norm.ppf(alpha/100.0)*forecast_std_egarch
    return egarch_VaR


def calculate_ARCH_VAR(portfolio_returns, window_length, alpha):
    forecast_mean_arch, forecast_std_arch = forecast_std(portfolio_returns, window_length, volatility_model='ARCH')
    fiarch_VaR = forecast_mean_arch + norm.ppf(alpha/100.0)*forecast_std_arch
    return fiarch_VaR


def calculate_HARCH_VAR(portfolio_returns, window_length, alpha):
    forecast_mean_harch, forecast_std_harch = forecast_std(portfolio_returns, window_length, volatility_model='HARCH')
    harch_VaR = forecast_mean_harch + norm.ppf(alpha/100.0)*forecast_std_harch
    return harch_VaR


def calculate_TARCH_VAR(portfolio_returns, window_length, alpha):
    forecast_mean_tarch, forecast_std_tarch = forecast_std(portfolio_returns, window_length, volatility_model='TARCH')
    harch_VaR = forecast_mean_tarch + norm.ppf(alpha/100.0)*forecast_std_tarch
    return harch_VaR


def calculate_GJR_GARCH_VAR(portfolio_returns, window_length, alpha):
    forecast_mean_tarch, forecast_std_tarch = forecast_std(portfolio_returns, window_length, volatility_model='GJR-GARCH')
    harch_VaR = forecast_mean_tarch + norm.ppf(alpha/100.0)*forecast_std_tarch
    return harch_VaR


def calculate_Historical_VAR(portfolio_returns, window_length, alpha):
    test_len = portfolio_returns.shape[0] - window_length
    hist_VaR = np.zeros((test_len, 1))
    for j in range(test_len):
        window = portfolio_returns[j:j + window_length]
        hist_VaR[j] = np.percentile(window, alpha)
    return hist_VaR


def calculate_MonteCarlo_VAR(portfolio_returns, window_length, alpha):
    seed(0)
    n_samples = 10000
    forecast_mean_arima = forecast_mean(portfolio_returns, window_length)
    forecast_std_garch = forecast_std(portfolio_returns, window_length, volatility_model='garch')
    STD = (np.sqrt(forecast_std_garch) / 100.0).reshape((-1, 1))
    MEAN = (forecast_mean_arima / 100.0).reshape((-1, 1))
    raw_samples = np.random.normal(0, 1, size=(1, n_samples))
    samples_with_scale = np.repeat(MEAN, n_samples, axis=1) + np.dot(STD, raw_samples)
    mc_VaR = np.percentile(samples_with_scale, q=alpha, axis=1)
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


def calculate_ratios(portfolio_returns, window_length, alpha):
    test_returns = portfolio_returns[window_length:]
    var_models_pm = pd.DataFrame(columns=['name', 'PM', 'ratio'])
    var_models = pd.DataFrame()
    #######################################################################################################
    # var_models['CAViaR_Sym'] = calculate_CAViaR_Sym_VAR(portfolio_returns, window_length, alpha)
    # var_models['CAViaR_Asym'] = calculate_CAViaR_Asym_VAR(portfolio_returns, window_length, alpha)
    # var_models['CAViaR_indirect_GARCH'] = calculate_CAViaR_indirect_GARCH_VAR(portfolio_returns, window_length, alpha)
    # var_models['CAViaR_adaptive'] = calculate_CAViaR_adaptive_VAR(portfolio_returns, window_length, alpha)
    var_models['Var_Covar'] = calculate_Var_Covar_VAR(portfolio_returns, window_length, alpha)
    var_models['RiskMetrics'] = calculate_RiskMetrics_VAR(portfolio_returns, window_length, alpha)
    var_models['GARCH'] = calculate_GARCH_VAR(portfolio_returns, window_length, alpha)
    var_models['FIGARCH'] = calculate_FIGARCH_VAR(portfolio_returns, window_length, alpha)
    var_models['EGARCH'] = calculate_EGARCH_VAR(portfolio_returns, window_length, alpha)
    var_models['ARCH'] = calculate_ARCH_VAR(portfolio_returns, window_length, alpha)
    var_models['HARCH'] = calculate_HARCH_VAR(portfolio_returns, window_length, alpha)
    var_models['TARCH'] = calculate_TARCH_VAR(portfolio_returns, window_length, alpha)
    var_models['GJR_GARCH'] = calculate_GJR_GARCH_VAR(portfolio_returns, window_length, alpha)
    var_models['Historical'] = calculate_Historical_VAR(portfolio_returns, window_length, alpha)
    var_models['MonteCarlo'] = calculate_MonteCarlo_VAR(portfolio_returns, window_length, alpha)
    # var_models['EVT'] = calculate_EVT_VAR(portfolio_returns, window_length, alpha)
    #######################################################################################################
    # plot(test_returns, var_models['CAViaR_Sym'].values, file_name='1.CAViaR_Sym')
    # plot(test_returns, var_models['CAViaR_Asym'].values, file_name='2.CAViaR_Asym')
    # plot(test_returns, var_models['CAViaR_indirect_GARCH'].values, file_name='3.CAViaR_indirect_GARCH')
    # plot(test_returns, var_models['CAViaR_adaptive'].values, file_name='4.CAViaR_adaptive')
    plot(test_returns, var_models['Var_Covar'].values, file_name='5.Var_Covar')
    plot(test_returns, var_models['RiskMetrics'].values, file_name='6.RiskMetrics')
    plot(test_returns, var_models['GARCH'].values, file_name='7.GARCH')
    plot(test_returns, var_models['FIGARCH'].values, file_name='8.FIGARCH')
    plot(test_returns, var_models['EGARCH'].values, file_name='9.EGARCH')
    plot(test_returns, var_models['ARCH'].values, file_name='10.ARCH')
    plot(test_returns, var_models['HARCH'].values, file_name='11.HARCH')
    plot(test_returns, var_models['TARCH'].values, file_name='12.TARCH')
    plot(test_returns, var_models['GJR_GARCH'].values, file_name='13.GJR_GARCH')
    plot(test_returns, var_models['Historical'].values, file_name='14.Historical')
    plot(test_returns, var_models['MonteCarlo'].values, file_name='15.MonteCarlo')
    # plot(test_returns, var_models['EVT'].values, file_name='16.EVT.png')
    #######################################################################################################
    # var_models_pm = var_models_pm.append({'name': 'CAViaR_Sym', 'PM': penalization_measure(test_returns, var_models['CAViaR_Sym'], alpha)}, ignore_index=True)
    # var_models_pm = var_models_pm.append({'name': 'CAViaR_Asym', 'PM': penalization_measure(test_returns, var_models['CAViaR_Asym'], alpha)}, ignore_index=True)
    # var_models_pm = var_models_pm.append({'name': 'CAViaR_indirect_GARCH', 'PM': penalization_measure(test_returns, var_models['CAViaR_indirect_GARCH'], alpha)}, ignore_index=True)
    # var_models_pm = var_models_pm.append({'name': 'CAViaR_adaptive', 'PM': penalization_measure(test_returns, var_models['CAViaR_adaptive'], alpha)}, ignore_index=True)
    var_models_pm = var_models_pm.append({'name': 'Var_Covar', 'PM': penalization_measure(test_returns, var_models['Var_Covar'], alpha)}, ignore_index=True)
    var_models_pm = var_models_pm.append({'name': 'RiskMetrics', 'PM': penalization_measure(test_returns, var_models['RiskMetrics'], alpha)}, ignore_index=True)
    var_models_pm = var_models_pm.append({'name': 'GARCH', 'PM': penalization_measure(test_returns, var_models['GARCH'], alpha)}, ignore_index=True)
    var_models_pm = var_models_pm.append({'name': 'FIGARCH', 'PM': penalization_measure(test_returns, var_models['FIGARCH'], alpha)}, ignore_index=True)
    var_models_pm = var_models_pm.append({'name': 'EGARCH', 'PM': penalization_measure(test_returns, var_models['EGARCH'], alpha)}, ignore_index=True)
    var_models_pm = var_models_pm.append({'name': 'ARCH', 'PM': penalization_measure(test_returns, var_models['ARCH'], alpha)}, ignore_index=True)
    var_models_pm = var_models_pm.append({'name': 'HARCH', 'PM': penalization_measure(test_returns, var_models['HARCH'], alpha)}, ignore_index=True)
    var_models_pm = var_models_pm.append({'name': 'TARCH', 'PM': penalization_measure(test_returns, var_models['TARCH'], alpha)}, ignore_index=True)
    var_models_pm = var_models_pm.append({'name': 'GJR_GARCH', 'PM': penalization_measure(test_returns, var_models['GJR_GARCH'], alpha)}, ignore_index=True)
    var_models_pm = var_models_pm.append({'name': 'Historical', 'PM': penalization_measure(test_returns, var_models['Historical'], alpha)}, ignore_index=True)
    var_models_pm = var_models_pm.append({'name': 'MonteCarlo', 'PM': penalization_measure(test_returns, var_models['MonteCarlo'], alpha)}, ignore_index=True)
    # var_models_pm = var_models_pm.append({'name': 'EVT', 'PM': penalization_measure(test_returns, var_models['EVT'], alpha)}, ignore_index=True)
    #######################################################################################################
    var_models_pm['ratio'] = var_models_pm['PM'] / sum(var_models_pm['PM'])
    return var_models, var_models_pm


def predictive_ability_test(test_returns, var_models, benchmark='GARCH'):
    var_models_error = var_models.subtract(test_returns, axis=0)
    # loss is a function of errors, it can be abs or power of 2
    var_models_loss = np.sqrt(np.power(var_models_error, 2))
    # benchmark_loss = var_models_loss[benchmark]
    # var_models_loss_wo_benchmark = var_models_loss.drop(columns=benchmark)
    # kappa_1 = var_models_loss_wo_benchmark.subtract(benchmark_loss, axis=0)
    kappa = var_models_loss.div(np.sum(var_models_loss, axis=1), axis=0)
    H_0 = np.mean(kappa, axis=0)
    W = np.sum(kappa > (1/var_models_loss.shape[1]))
    p = 0.5
    T = var_models.shape[0]
    W_hat = (W - p*T)/np.sqrt(p*(1-p)*T)
    return 0


def plot(returns, VARs, file_name=None):

    # Re-add the time series index
    r = pd.Series(returns)
    q = pd.Series(VARs)

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
        plt.savefig(file_name+'.png', bbox_inches="tight")
    plt.close("all")


def progressBar(value, end_value, bar_length=20):
    percent = float(value) / end_value
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rCompleted: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()
#import os
#directory = os.getcwd()+'/data' #"C:\\Users\\ehsan\Desktop\Torronto Stocks Data\\"
#rets = pd.read_csv(directory + "/returns_TSX.csv", header=0)
#rets['date'] = pd.to_datetime(rets['date'])
# rets = rets.set_index('date')
#n, m = rets.shape
#weights = np.ones((m, 1)) / m
#portfolio_returns = np.dot(rets, weights).squeeze()
#window_length = np.floor(0.875 * n).astype('int64')
# alpha = 5
# var_models, var_models_pm = calculate_ratios(portfolio_returns, window_length, alpha)

# evar = pd.read_csv('evar.csv')
# evar['date'] = pd.to_datetime(evar['date'])
# evar = evar.set_index('date')
# var_models['evar'] = pd.Series(evar['0'])

# plot(test_returns, var_models['evar'].values, file_name='1.eVaR')