# Funzioni Assigment 5.0
import pandas as pd
import numpy as np
import scipy
import math
from scipy.stats import norm


def AnalyticalNormalMeasures(alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay, returns):
    # We calculate the loss as the portfolio's value in t multiplied by the scalar product between wt and xt
    Loss = - portfolioValue * (returns.dot(weights))  # np_array.dot computes the product between matrices
    Loss_mean = np.mean(Loss)  # We calculate the mean of losses
    Loss_std = np.std(Loss)  # We calculate the standard deviation of losses
    VaR_std = norm.ppf(alpha)  # VaR of a std normal is the inverse of the cdf evaluated in alpha (N^(-1)(alpha))
    VaR = riskMeasureTimeIntervalInDay * Loss_mean + math.sqrt(
        riskMeasureTimeIntervalInDay) * Loss_std * VaR_std  # VaR expressed in function of VaR_std with delta=current
    # time window (1 in this case)
    # ES for a std normal is the pdf of a normal evaluated in the VaR of a std normal
    ES_std = norm.pdf(VaR_std) / (1 - alpha)
    ES = riskMeasureTimeIntervalInDay * Loss_mean + math.sqrt(
        riskMeasureTimeIntervalInDay) * Loss_std * ES_std  # ES expressed in function of ES_std
    return VaR, ES


def read_our_CSV(df):
    df = df.fillna(method='ffill')  # we fill the missing values of the stocks with the previous known ones

    df_ptf = df['2016-03-21':'2019-03-19'] # we selected 3y from 20-03-2019 bckw. up to 22-03-2016
    df_ptf = df_ptf.loc[:, ['ADSGn.DE', 'ALVG.DE', 'MUVGn.DE','OREP.PA']]  # we select only the 4 columns corresponding to Adidas, Allianz, Munich RE and l'Oreal
    df_den = df['2016-03-22':'2019-03-20']  # we selected 3y from one day before 20-03-2019 (19-03-2019) up one day before the last date of df_ptf (21-03-2016)
    df_den = df_den.loc[:, ['ADSGn.DE', 'ALVG.DE', 'MUVGn.DE', 'OREP.PA']]

    # we pass to numpy arrays to perform the logarithm
    np_den = df_den.to_numpy()
    np_num = df_ptf.to_numpy()
    return np_num, np_den
