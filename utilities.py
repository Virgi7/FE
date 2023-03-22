# Funzioni Assigment 5.0
import pandas as pd
import numpy as np
import scipy
import math
from scipy.stats import norm


def AnalyticalNormalMeasures(alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay, returns):
    # We calculate the loss as the portfolio's value in t multiplied by the scalar product between wt and xt
    Loss = -portfolioValue * (returns.dot(weights))  # np_array.dot computes the product between matrices
    Loss_mean = np.mean(Loss)  # We calculate the mean of losses
    Loss_std = np.std(Loss)  # We calculate the standard deviation of losses
    VaR_std = norm.ppf(alpha)  # VaR of a std normal is the inverse of the cdf evaluated in alpha (N^(-1)(alpha))
    VaR = riskMeasureTimeIntervalInDay * Loss_mean + math.sqrt(riskMeasureTimeIntervalInDay) * Loss_std * VaR_std  # VaR expressed in function of VaR_std with delta=current
    # time window (1 in this case)
    ES_std = norm.pdf(VaR_std) / (1-alpha)  # ES for a std normal is the pdf of a normal evaluated in the VaR of a std normal
    ES = riskMeasureTimeIntervalInDay * Loss_mean + math.sqrt(riskMeasureTimeIntervalInDay) * Loss_std * ES_std  # ES expressed in function of ES_std
    return VaR, ES


def read_our_CSV(df, name_stocks, dates_num, dates_den):
    df.fillna(method='ffill', inplace=True)  # we fill the missing values of the stocks with the previous known ones

    # We select stocks properly in order to perform the right computation of the returns
    df_ptf = df[dates_num[0]:dates_num[1]]  # we selected 3y from 20-03-2019 bckw. up to 22-03-2016

    df_ptf = df_ptf.loc[:, name_stocks]  # we select only the 4 columns corresponding to Adidas, Allianz, Munich RE
    # and l'Oreal
    df_den = df[dates_den[0]:dates_den[1]]  # we selected 3y from one day before 20-03-2019 (19-03-2019) up one day before the last date
    # of df_ptf (21-03-2016)
    df_den = df_den.loc[:, name_stocks]

    # we pass to numpy arrays to perform the logarithm
    np_den = df_den.to_numpy()
    np_num = df_ptf.to_numpy()
    return np_num, np_den

def plausibilityCheck(returns, portfolioWeights, alpha, portfolioValue, riskMeasureTimeIntervalInDay):
    l = np.zeros((len(portfolioWeights), 1))
    u = np.zeros((len(portfolioWeights), 1))
    sens = np.zeros((len(portfolioWeights), 1))
    av = np.zeros((len(portfolioWeights), 1))
    sVaR = np.zeros((len(portfolioWeights), 1))
    C=np.zeros((len(portfolioWeights), len(portfolioWeights)))
    samples = int(returns.shape[0] - riskMeasureTimeIntervalInDay + 1)
    added_returns = np.zeros((samples, returns.shape[1]))
    for i in range(samples):
        for j in range(i, (i + riskMeasureTimeIntervalInDay)):
            added_returns[samples - 1 - i, :] = added_returns[samples - 1 - i, :] + returns[returns.shape[0] - 1 - j, :]
    for i in range (len(portfolioWeights)):
        l[i]=np.quantile(added_returns[:,i],1-alpha)
        u[i]=np.quantile(added_returns[:,i],alpha)
        #print(l)
        sens[i] = portfolioValue * portfolioWeights[i]
        av[i]=(abs(l[i])+abs(u[i]))/2
        sVaR[i]=sens[i]*av[i]
    C=np.corrcoef(added_returns.T)
    print(type(sVaR), type(C))
    VaR=math.sqrt((sVaR.T).dot(C).dot(sVaR))
    return VaR



