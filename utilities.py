# Functions Assignment 5.0
import numpy as np
import pandas as pd
from scipy.stats import norm
from numpy import linalg
import math
import scipy.stats as st
import random


def AnalyticalNormalMeasures(alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay, returns):
    # We calculate the loss as the portfolio's value in t multiplied by the scalar product between wt and xt
    Loss = -portfolioValue * (returns.dot(weights))  # np_array.dot computes the product between matrices
    Loss_mean = np.mean(Loss)  # We calculate the mean of losses
    Loss_std = np.std(Loss)  # We calculate the standard deviation of losses
    VaR_std = norm.ppf(alpha)  # VaR of a std normal is the inverse of the cdf evaluated in alpha (N^(-1)(alpha))
    # VaR expressed in function of VaR_std with delta=current time window (1 in this case)
    VaR = riskMeasureTimeIntervalInDay * Loss_mean + math.sqrt(riskMeasureTimeIntervalInDay) * Loss_std * VaR_std
    ES_std = norm.pdf(VaR_std) / (1-alpha)  # ES for a std normal is the pdf of a normal evaluated in the VaR of a std normal
    # ES expressed in function of ES_std
    ES = riskMeasureTimeIntervalInDay * Loss_mean + math.sqrt(riskMeasureTimeIntervalInDay) * Loss_std * ES_std
    return VaR, ES


def plausibilityCheck(returns, portfolioWeights, alpha, portfolioValue, riskMeasureTimeIntervalInDay):
    l = np.zeros((len(portfolioWeights), 1))
    u = np.zeros((len(portfolioWeights), 1))
    sens = np.zeros((len(portfolioWeights), 1))
    av = np.zeros((len(portfolioWeights), 1))
    sVaR = np.zeros((len(portfolioWeights), 1))
    # number of returns we consider, we will add the returns over the corresponding time intervals
    added_returns = aggregateReturns(returns, riskMeasureTimeIntervalInDay)
    for i in range(len(portfolioWeights)):
        # lower quantile (of the i-th risk factor distribution)
        l[i] = np.quantile(added_returns[:, i], 1 - alpha)
        # upper quantile (of the i-th risk factor distribution)
        u[i] = np.quantile(added_returns[:, i], alpha)
        # sensitivity of the portfolio with respect to the i-th risk factor
        sens[i] = portfolioValue * portfolioWeights[i]
        # mean of the upper and lower quantile (in absolute value)
        av[i] = (abs(l[i])+abs(u[i]))/2
        # stressed VaR (with respect to the i-th risk factor)
        sVaR[i] = sens[i]*av[i]
    # Correlation matrix
    C = np.corrcoef(added_returns.T)
    # VaR
    VaR = math.sqrt(sVaR.T.dot(C).dot(sVaR))
    return VaR


def HSMeasurements(returns, alpha, weights, portfolioValue, RiskMeasureTimeIntervalInDay):
    added_returns = aggregateReturns(returns, RiskMeasureTimeIntervalInDay)
    # linearized loss of the portfolio, there is no cost term
    loss = - portfolioValue * added_returns.dot(weights)
    # we order the losses in decreasing order
    loss_sorted = sorted(loss, reverse=True)
    # VaR as the 1 - alpha quantile of the loss distribution
    VaR = np.quantile(loss_sorted, alpha)
    # ES as the mean of the losses greater than the VaR
    ES = np.mean(loss_sorted[0:int(math.floor((added_returns.shape[0]-1) * (1-alpha))) + 1])
    return ES, VaR


def WHSMeasurements(returns, alpha, Lambda, weights, portfolioValue, RiskMeasureTimeIntervalInDay):
    added_returns = aggregateReturns(returns, RiskMeasureTimeIntervalInDay)
    # weights of the Historical Simulation
    lambdas = WHSweights(Lambda, added_returns.shape[0])
    # linearized loss of the portfolio multiplied by the weights of the WHS
    loss = -portfolioValue * added_returns.dot(weights)
    # We sort the weights in a way that they correspond to the ordered losses
    all_sorted = sort_as(loss.reshape(len(loss)), lambdas.reshape(len(loss)))
    loss_sorted = all_sorted[:, 0]
    lambdas_sorted = all_sorted[:, 1]
    # we find the greatest i such that sum(lambdas[i:end]) <= 1 - alpha
    i = searchLevel(lambdas_sorted, alpha)
    # Var as the i-th loss
    VaR = float(loss_sorted[i])
    # ES as average of losses greater than the VaR
    ES = float(sum(loss_sorted[0:i] * lambdas_sorted[0:i]) / sum(lambdas_sorted[0:i]))
    return ES, VaR


def PrincCompAnalysis(yearlyCovariance, yearlyMeanReturns, weights, H, alpha, numberOfPrincipalComponents,
                      portfolioValue):
    # spectral decomposition of the variance covariance matrix
    eigenvalues, eigenvectors = linalg.eig(yearlyCovariance)
    # we order the set of eigenvalues
    all_sorted = sort_as(eigenvalues, yearlyMeanReturns)
    eigenvalues_sorted = all_sorted[:, 0]
    mean_sorted = all_sorted[:, 1]
    weights_sorted = weights
    gamma = np.zeros((len(eigenvalues), len(eigenvalues)))
    for i in range(len(eigenvalues)):
        all_sorted1 = sort_as(eigenvalues, eigenvectors[i, :])
        gamma[i, :] = all_sorted1[:, 1]
    # Projected weights
    weights_hat = gamma.T.dot(weights_sorted)
    # Projected mean vector
    mean_hat = gamma.T.dot(mean_sorted.reshape(len(mean_sorted), 1))
    # reduced standard deviation
    sigma_red = (H * (weights_hat[0: numberOfPrincipalComponents] ** 2).T.dot(eigenvalues_sorted[0: numberOfPrincipalComponents])) ** (1 / 2)
    # reduced mean
    mean_red = H * sum(mean_hat[0: numberOfPrincipalComponents] * weights_hat[0: numberOfPrincipalComponents])
    # VaR and ES with the usual formulas
    VaR = float(portfolioValue * (- mean_red + sigma_red * st.norm.ppf(alpha)))
    ES = float(portfolioValue * (- mean_red + sigma_red * st.norm.pdf(st.norm.ppf(1 - alpha)) / (1 - alpha)))
    return ES, VaR


def bootstrapStatistical(numberOfSamplesToBootstrap, returns):
    random.seed(5)
    # number of risk factors
    n = returns.shape[0]
    # we initialize the output
    samples = np.zeros((numberOfSamplesToBootstrap, returns.shape[1]))
    for i in range(numberOfSamplesToBootstrap):
        # we extract which risk factor use for the simulation
        x = int(random.randint(0, n-1))
        # i-th simulated risk measure
        samples[i] = returns[x, :]
    return samples


def FullMonteCarloVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,
                      volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears):
    # length of the time interval
    delta = int(math.floor(riskMeasureTimeIntervalInYears * NumberOfDaysPerYears))
    # number of returns we consider, we will add the returns over the corresponding time intervals
    added_returns = logReturns * delta ** (1/2)
    # simulated stock price
    simulated_stock = stockPrice * np.exp(added_returns)
    # Time to maturity of the put options minus the delta in years
    TTM_simulated = timeToMaturityInYears - riskMeasureTimeIntervalInYears
    simulated_put = np.zeros((len(simulated_stock), 1))
    for i in range(len(simulated_stock)):
        # B&S formula applied to the simulated stock price
        simulated_put[i] = BS_PUT(simulated_stock[i], strike, TTM_simulated, rate, dividend, volatility)
    # price today of the put option
    putPrice = BS_PUT(stockPrice, strike, timeToMaturityInYears, rate, dividend, volatility)
    # simulated losses
    loss = - numberOfShares * (simulated_stock - stockPrice * np.ones((len(simulated_stock), 1))) - numberOfPuts * (simulated_put - putPrice * np.ones((len(simulated_put), 1)))
    loss_sorted = sorted(loss, reverse=True)
    # VaR as the 1 - alpha quantile of the loss distribution
    VaR = np.quantile(loss_sorted, alpha)
    return VaR


def DeltaNormalVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,
                   volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears):
    # length of the time interval
    delta = int(math.floor(riskMeasureTimeIntervalInYears * NumberOfDaysPerYears))
    # number of returns we consider, we will add the returns over the corresponding time intervals
    added_returns = logReturns * delta ** (1/2)
    # sensitivity of our portfolio
    sens = BS_PUT_delta(stockPrice, strike, timeToMaturityInYears, rate, dividend, volatility)
    # simulated linearized losses
    loss = - numberOfPuts * stockPrice * sens * added_returns - numberOfShares * stockPrice * added_returns
    loss_sorted = sorted(loss, reverse=True)
    # VaR as the 1 - alpha quantile of the loss distribution
    VaR = np.quantile(loss_sorted, alpha)
    return VaR


def DeltaGammaNormalVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,
                        volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears):
    # length of the time interval
    delta = int(math.floor(riskMeasureTimeIntervalInYears * NumberOfDaysPerYears))
    # number of returns we consider, we will add the returns over the corresponding time intervals
    added_returns = logReturns * delta ** (1/2)
    # sensitivity of our portfolio
    sens = BS_PUT_delta(stockPrice, strike, timeToMaturityInYears, rate, dividend, volatility)
    gamma = BS_PUT_gamma(stockPrice, strike, timeToMaturityInYears, rate, dividend, volatility)
    # simulated linearized losses
    loss = - numberOfPuts * stockPrice * (sens * added_returns + (1/2) * gamma * stockPrice * (added_returns ** 2)) - numberOfShares * stockPrice * added_returns
    loss_sorted = sorted(loss, reverse=True)
    # VaR as the 1 - alpha quantile of the loss distribution
    VaR = np.quantile(loss_sorted, alpha)
    return VaR


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


def sort_as(a, b):
    my_array = np.array([a, b])
    df = pd.DataFrame(my_array).T
    df = df.rename(columns={0: "a", 1: "b"})
    df_sorted = df.sort_values(by='a', ascending=False)
    array_1 = df_sorted.to_numpy()
    return array_1


def searchLevel(weights, alpha):
    # we find the greatest i such that sum(lambdas[i:end]) <= 1 - alpha
    i = -1
    weights_sum = 0
    while weights_sum <= (1 - alpha):
        i += 1
        weights_sum += weights[i]
    return i


def tree_gen(sigma, steps, DF, S0, delta, T):  # T è la maturity
    u = math.exp(sigma*math.sqrt(delta/steps))  # Delta = 1 year/n = number of steps for each year
    d = math.exp(-sigma*math.sqrt(delta/steps))
    tree = np.zeros((int(steps*T/delta + 1), int(steps*T/delta + 1)))
    tree[0][0] = S0
    for i in range(1, int(steps*T/delta) + 1):
        for j in range(i + 1):
            tree[j][i] = S0 * (u ** (i - j)) * (d ** j)
    return tree[:, range(steps, steps * T + 1, steps)] / DF


def priceCliquetTree(S0, disc, tree, steps, sigma, rec, SurProb, datesInYears):
    # up and down in the tree
    u = math.exp(sigma * math.sqrt(1 / steps))
    d = math.exp(-sigma * math.sqrt(1 / steps))
    # probability of up
    q = (1 - d)/(u - d)
    # Survival probabilities for the expires in datesInYears
    e_function = (SurProb[0: len(SurProb) - 1] - SurProb[1: len(SurProb)]) * disc[0: len(SurProb) - 1]
    B_bar = SurProb[1: len(SurProb)] * disc[0: len(SurProb) - 1]
    T = len(datesInYears)
    payoff = np.zeros(tree.shape)
    # we consider the payments as payoff of ATM call options, the premium computed with B&S formula
    rate = -np.log(disc[1]) / datesInYears[0]
    payoff[0, 0] = BS_CALL(S0, S0, datesInYears[0], rate, 0, sigma)
    for i in range(1, T):
        TTM = datesInYears[i] - datesInYears[i - 1]
        rate = -np.log(disc[i + 1] / disc[i]) / TTM
        for j in range(i * steps + 1):
            payoff[j, i] = BS_CALL(tree[j, i-1], tree[j, i-1], TTM, rate, 0, sigma) * bincoeff(i * steps, j) * (q ** (i * steps - j)) * ((1 - q) ** j)
    # We multiply by the discounts, the survival probabilities and the recovery multiplied by the default probability in each time interval
    price = payoff * (B_bar + rec * e_function)
    price = sum(sum(price))
    return price


def priceCliquetBS(S0, disc, h, sigma, rec, SurProb, datesInYears):
    # Survival probabilities for the expires in datesInYears
    e_function = (SurProb[0: len(SurProb) - 1] - SurProb[1: len(SurProb)]) * disc[0: len(SurProb) - 1]
    B_bar = SurProb[1: len(SurProb)] * disc[0: len(SurProb) - 1]
    T = len(datesInYears)
    payoff = np.zeros((T, 1))
    # we consider the payments as payoff of ATM call options, the premium computed with B&S formula
    # then we compute the expectation with respect to the future stock price
    for i in range(T):
        TTM = datesInYears[i] - datesInYears[i - 1]
        if i == 0:
            TTM = datesInYears[0]
            rate = -np.log(disc[1]) / TTM
            payoff[0] = BS_CALL(S0, S0, TTM, rate, 0, sigma)
        else:
            rate = -np.log(disc[i + 1] / disc[i]) / TTM
            y = -6
            S_1 = S0 * np.exp(- (sigma ** 2 / 2) * datesInYears[i] + sigma * np.sqrt(datesInYears[i]) * (y - h)) / disc[i]
            while y <= 6:
                S = S0 * np.exp(- (sigma ** 2 / 2) * datesInYears[i] + sigma * np.sqrt(datesInYears[i]) * y) / disc[i]
                payoff[i] += (BS_CALL(S, S, TTM, rate, 0, sigma) * st.norm.pdf(y) + BS_CALL(S_1, S_1, TTM, rate, 0, sigma) * st.norm.pdf(y - h)) * h / 2
                S_1 = S
                y = y + h
    # We multiply by the discounts, the survival probabilities and the recovery multiplied by the default probability in each time interval
    price = float((B_bar + rec * e_function).dot(payoff))
    return price


def priceCliquetMC(S0, disc, N, M, sigma, rec, SurProb, datesInYears):
    # Survival probabilities for the expires in datesInYears
    e_function = (SurProb[0: len(SurProb) - 1] - SurProb[1: len(SurProb)]) * disc[1: len(SurProb)]
    B_bar = SurProb[1: len(SurProb)] * disc[1: len(SurProb)]
    T = len(datesInYears) - 1
    payoff = np.zeros((T, 1))
    # we consider the payments as payoff of ATM call options, the premium computed with B&S formula
    # then we compute the expectation with respect to the future stock price
    S = GBMsimulation(N, S0, disc, sigma, datesInYears[T], M)
    for i in range(T):
        payoff[i] = np.mean((S[:, i + 1] - S[:, i]) * (S[:, i + 1] >= S[:, i]))
    # We multiply by the discounts, the survival probabilities and the recovery multiplied by the default probability in each time interval
    price = float((B_bar + rec * e_function).dot(payoff))
    return price


def bincoeff(n, k):
    if k == 0 or k == n:
        coeff = 1
    else:
        if k > n:
            coeff = 0
        else:
            a = math.factorial(n)
            b = math.factorial(k)
            c = math.factorial(n - k)
            coeff = a / (b * c)
    return coeff


def BS_PUT(S, K, T, r, d, sigma):
    # B&S formula for a put option
    d1 = (np.log(S / K) + (r - d + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * st.norm.cdf(-d2) - S * np.exp(-d * T) * st.norm.cdf(-d1)


def BS_PUT_delta(S, K, T, r, d, sigma):
    # B&S formula for a put option
    d1 = (np.log(S / K) + (r - d + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    return - np.exp(-d * T) * st.norm.cdf(-d1)  # Derivative w.r.t. the stock


def BS_PUT_gamma(S, K, T, r, d, sigma):
    # B&S formula for a put option
    d1 = (np.log(S / K) + (r - d + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-d * T) * st.norm.pdf(-d1) / (S * sigma * np.sqrt(T))  # Derivative w.r.t. the stock


def BS_CALL(S, K, T, r, d, sigma):
    # B&S formula for a put option
    d1 = (np.log(S / K) + (r - d + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-d * T) * st.norm.cdf(d1) - K * np.exp(-r * T) * st.norm.cdf(d2)


def GBMsimulation(N, S0, DF, sigma, T, M):
    np.random.seed(5)
    # length of the time step
    dt = T / M
    S = S0 * np.ones((N, M + 1))
    W = np.zeros((N, M + 1))
    for j in range(1, M + 1):
        W[:, j] = W[:, j - 1] + np.sqrt(dt) * np.random.normal(0, 1, N)
        S[:, j] = S0 * np.exp(- (sigma ** 2 / 2) * dt * j + sigma * W[:, j])
    S = S[:, range(0, M + 1, int(np.floor(M / T)))]
    for i in range(N):
        S[i, :] = S[i, :] / DF
    return S
