# Functions Assignment 5.0
import numpy as np


def read_our_CSV(df, name_stocks, dates_num, dates_den):
    # we fill the missing values of the stocks with the previous known ones
    df.fillna(method='ffill', inplace=True)
    # We select stocks properly in order to perform the right computation of the returns
    df_ptf = df[dates_num[0]:dates_num[1]]  # we selected 3y from 20-03-2019 bckwd up to 22-03-2016
    df_ptf = df_ptf.loc[:, name_stocks]  # we select only the 4 columns corresponding to Adidas, Allianz, Munich RE
    # and l'Oreal
    # we selected 3y from one day before 20-03-2019 (19-03-2019) up one day before the last date of df_ptf (21-03-2016)
    df_den = df[dates_den[0]:dates_den[1]]
    df_den = df_den.loc[:, name_stocks]
    # we pass to numpy arrays to perform the logarithm
    np_den = df_den.to_numpy()
    np_num = df_ptf.to_numpy()
    return np_num, np_den


def ZeroRate(dates, discounts, date):
    # dates is in days
    # discounts is the set of corresponding discount factors at given dates
    # date is the date where we want to know the discount factor is in years
    datesyears = (dates[1::] - dates[0]) / 365
    zeros = np.log(discounts[1::]) / datesyears
    return np.interp(date, datesyears, zeros)


def aggregateReturns(returns, delta):
    # Number of returns we consider
    samples = int(returns.shape[0] / delta)
    added_returns = np.zeros((samples, returns.shape[1]))
    # We add the returns in each range (not overlapped)
    for i in range(samples):
        for j in range(i * delta, ((i + 1) * delta)):
            # we add the returns over the time interval [i, i + RiskMeasureTimeIntervalInDay]
            added_returns[i, :] += returns[j, :]
    return added_returns


def WHSweights(Lambda, n):
    # normalization constant
    C = (1 - Lambda) / (1 - Lambda ** n)
    # weights of the Weighted Historical Simulation
    lambdas = np.zeros((n, 1))
    for i in range(n):
        lambdas[i] = C * Lambda ** i
    return lambdas


def sort_as(a, a_sorted, b):
    # Gives a version of b sorted as a_sorted
    b_sorted = b
    for i in range(len(a_sorted)):
        # we order the weights of the WHS following the order of the losses
        b_sorted[i] = b[a.tolist().index(a_sorted[i])]
    return b_sorted


def searchLevel(weights, alpha):
    # we find the greatest i such that sum(lambdas[i:end]) <= 1 - alpha
    i = -1
    weights_sum = 0
    while weights_sum <= (1 - alpha):
        i += 1
        weights_sum += weights[i]
    return i
