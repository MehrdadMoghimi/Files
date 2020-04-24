# -*- coding: utf-8 -*-
"""
Created on Thu May 10 13:18:10 2018

@author: zhli6157
"""

import pandas as pd
import numpy as np
from scipy import stats


# ==============================================================================
# Kupiec Uncondition Coverage Backtesting, Proportion of Failures(POF)
# Defined as UCoverage
# UCoverage(Returns, Value at Risk, Confidence Level of VaR)
# ==============================================================================

def UCoverage(Returns, VaR, ConfidenceLevel):
    Compare = pd.concat([Returns, VaR], axis=1)
    Number_of_Fail = len(Compare[Compare.iloc[:, 0] < Compare.iloc[:, 1]])
    N = Number_of_Fail
    T = len(Compare)
    t = (1 - N / T) ** (T - N) * (N / T) ** N
    c = ((ConfidenceLevel) ** (T - N)) * ((1 - ConfidenceLevel) ** N)
    Likelihood_Ratio = 2 * np.log(t) - 2 * np.log(c)
    return Likelihood_Ratio, 1-stats.chi2.cdf(Likelihood_Ratio, 1)


# ==============================================================================
def FailRate(Returns, VaR):
    Compare = pd.concat([Returns, VaR], axis=1)
    Number_of_Fail = len(Compare[Compare.iloc[:, 0] < Compare.iloc[:, 1]])
    N = Number_of_Fail
    T = len(Compare)
    FailRate = N / T
    return FailRate


# ==============================================================================
# Christoffersen's Interval Forecast Tests, Conditional Coverage Backtesting
# Defined as LRCCI
# LRCCI(Returns, Value at Risk, Confidence Level of VaR)
# ==============================================================================
def LRCCI(Returns, VaR):
    LRCC = pd.concat([Returns, VaR], axis=1)
    TF = LRCC.iloc[:, 0] > LRCC.iloc[:, 1]
    n00 = 0
    n10 = 0
    n01 = 0
    n11 = 0
    for i in range(len(TF) - 1):
        if TF[i] == True and TF[i + 1] == True:
            n00 = n00 + 1
    for m in range(len(TF) - 1):
        if TF[m] == False and TF[m + 1] == True:
            n10 = n10 + 1
    for q in range(len(TF) - 1):
        if TF[q] == True and TF[q + 1] == False:
            n01 = n01 + 1
    for f in range(len(TF) - 1):
        if TF[f] == False and TF[f + 1] == False:
            n11 = n11 + 1

    pi0 = n01 / (n00 + n01)
    pi1 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    Numeritor = ((1 - pi) ** (n00 + n10)) * (pi ** (n01 + n11))
    Denominator = ((1 - pi0) ** (n00)) * (pi0 ** n01) * ((1 - pi1) ** (n10)) * (pi1 ** n11)
    LRCCI = -2 * np.log(Numeritor / Denominator)
    return LRCCI, 1-stats.chi2.cdf(LRCCI, 1)


# ==============================================================================
# Regulator's Loss Function Family
# Mathmatical Reference: The role of the loss function in value-at-risk comparisons
# The score for the complete sample is the sum of each individual point
# ==============================================================================
# Lopez's quadratic (RQL)
# Defined as RQL
# RQL(Returns, Value at Risk)
# ==============================================================================
def RQL(Returns, VaR):
    Compare = pd.concat([Returns, VaR], axis=1)
    Compare = Compare[Compare.iloc[:, 0] < Compare.iloc[:, 1]]
    quadratic = 1 + (Compare.iloc[:, 1] - Compare.iloc[:, 0]) ** 2
    RQL_mean = np.mean(quadratic)
    RQL_sum = np.sum(quadratic)
    return RQL_mean, RQL_sum


# ==============================================================================
# Linear (RL)
# Defined as RL
# RL(Returns, Value at Risk)
# ==============================================================================
def RL(Returns, VaR):
    Compare = pd.concat([Returns, VaR], axis=1)
    Compare = Compare[Compare.iloc[:, 0] < Compare.iloc[:, 1]]
    quadratic = (Compare.iloc[:, 1] - Compare.iloc[:, 0])
    RL_mean = np.mean(quadratic)
    RL_sum = np.sum(quadratic)
    return RL_mean, RL_sum


# ==============================================================================
# Quadratic (RQ)
# Defined as RQ
# RQ(Returns, Value at Risk)
# ==============================================================================
def RQ(Returns, VaR):
    Compare = pd.concat([Returns, VaR], axis=1)
    Compare = Compare[Compare.iloc[:, 0] < Compare.iloc[:, 1]]
    quadratic = (Compare.iloc[:, 1] - Compare.iloc[:, 0])**2
    RQ_mean = np.mean(quadratic)
    RQ_sum = np.sum(quadratic)
    return RQ_mean, RQ_sum


# ==============================================================================
# Caporin_1 (RC_1)
# Defined as RC_1
# RC_1(Returns, Value at Risk)
# ==============================================================================
def RC_1(Returns, VaR):
    Compare = pd.concat([Returns, VaR], axis=1)
    Compare = Compare[Compare.iloc[:, 0] < Compare.iloc[:, 1]]
    quadratic = np.abs(1-np.abs(Compare.iloc[:, 0]/Compare.iloc[:, 1]))
    RC1_mean = np.mean(quadratic)
    RC1_sum = np.sum(quadratic)
    return RC1_mean, RC1_sum


# ==============================================================================
# Caporin_2 (RC_2)
# Defined as RC_2
# RC_2(Returns, Value at Risk)
# ==============================================================================
def RC_2(Returns, VaR):
    Compare = pd.concat([Returns, VaR], axis=1)
    Compare = Compare[Compare.iloc[:, 0] < Compare.iloc[:, 1]]
    quadratic = (np.abs(Compare.iloc[:, 0]) - np.abs(Compare.iloc[:, 1])) ** 2 / (np.abs(Compare.iloc[:, 1]))
    RC2_mean = np.mean(quadratic)
    RC2_sum = np.sum(quadratic)
    return RC2_mean, RC2_sum


# ==============================================================================
# Caporin_3 (RC_3)
# Defined as RC_3
# RC_3(Returns, Value at Risk)
# ==============================================================================
def RC_3(Returns, VaR):
    Compare = pd.concat([Returns, VaR], axis=1)
    Compare = Compare[Compare.iloc[:, 0] < Compare.iloc[:, 1]]
    quadratic = (np.abs(Compare.iloc[:, 1] - Compare.iloc[:, 0]))
    RC3_mean = np.mean(quadratic)
    RC3_sum = np.sum(quadratic)
    return RC3_mean, RC3_sum


# ==============================================================================
# Firm's Loss Function Family
# Mathmatical Reference: The role of the loss function in value-at-risk comparisons
# ==============================================================================
# ==============================================================================
# Caporin_1 (FC_1)
# Defined as FC_1
# FC_1(Returns, Value at Risk)
# ==============================================================================
def FC_1(Returns, VaR):
    Compare = pd.concat([Returns, VaR], axis=1)
    quadratic = np.abs(1 - np.abs(Compare.iloc[:, 0] / Compare.iloc[:, 1]))
    FC1_mean = np.mean(quadratic)
    FC1_sum = np.sum(quadratic)
    return FC1_mean, FC1_sum


# ==============================================================================
# Caporin_2 (FC_2)
# Defined as FC_2
# FC_2(Returns, Value at Risk)
# ==============================================================================
def FC_2(Returns, VaR):
    Compare = pd.concat([Returns, VaR], axis=1)
    quadratic = (np.abs(Compare.iloc[:, 0]) - np.abs(Compare.iloc[:, 1]) ** 2) / np.abs(Compare.iloc[:, 1])
    FC2_mean = np.mean(quadratic)
    FC2_sum = np.sum(quadratic)
    return FC2_mean, FC2_sum


# ==============================================================================
# Caporin_3 (FC_3)
# Defined as FC_3
# FC_3(Returns, Value at Risk)
# ==============================================================================
def FC_3(Returns, VaR):
    Compare = pd.concat([Returns, VaR], axis=1)
    quadratic = np.abs(Compare.iloc[:, 1] - Compare.iloc[:, 0])
    FC3_mean = np.mean(quadratic)
    FC3_sum = np.sum(quadratic)
    return FC3_mean, FC3_sum


# ==============================================================================
# Quantile Loss Function
# Reference: The Use of GARCH Models in VaR Estimation.
# Defined as QL
# Ql(Returns, Value at Risk, Condidence Level of VaR)
# ==============================================================================
def QL(Returns, VaR, ConfidenceLevel):
    QL = []
    for i in range(len(VaR)):
        if Returns[i] < VaR[i]:
            QuantileLoss = (Returns[i] - VaR[i]) ** 2
        else:
            QuantileLoss = (Returns[i + 1:].quantile(1 - ConfidenceLevel) - VaR[i]) ** 2
        QL.append(QuantileLoss)
    QL_Score = np.sum(QL)
    return QL_Score
