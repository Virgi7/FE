import numpy as np
from numpy import linalg
import math
import scipy.stats as st
import random


def HSMeasurements(returns, alpha, weights, portfolioValue, RiskMeasureTimeIntervalInDay):
    # number of returns we consider, we will add the returns over the corresponding time intervals
    samples = int(returns.shape[0] - RiskMeasureTimeIntervalInDay + 1)
    added_returns = np.zeros((samples, returns.shape[1]))
    for i in range(samples):
        for j in range(i, (i + RiskMeasureTimeIntervalInDay)):
            # we add the returns over the time interval [i, i + RiskMeasureTimeIntervalInDay]
            added_returns[samples - 1 - i, :] = added_returns[samples - 1 - i, :] + returns[returns.shape[0] - 1 - j, :]
    # linearized loss of the portfolio, there is no cost term
    loss = - portfolioValue * added_returns.dot(weights)
    # we order the losses in decreasing order
    loss_sorted = sorted(loss, reverse=True)
    # VaR as the 1 - alpha quantile of the loss distribution
    VaR = float(loss_sorted[int(math.floor(samples-1) * (1-alpha))])
    # ES as the mean of the losses greater than the VaR
    ES = np.mean(loss_sorted[0:int(math.floor((samples-1) * (1-alpha)))])
    return ES, VaR


def WHSMeasurements(returns, alpha, Lambda, weights, portfolioValue, RiskMeasureTimeIntervalInDay):
    # number of returns we consider, we will add the returns over the corresponding time intervals
    samples = int(returns.shape[0] - RiskMeasureTimeIntervalInDay + 1)
    # normalization constant
    C = (1 - Lambda) / (1 - Lambda ** samples)
    # weights of the Historical Simulation
    lambdas = np.zeros((samples, 1))
    for i in range(samples):
        lambdas[i] = C * Lambda ** i
    added_returns = np.zeros((samples, returns.shape[1]))
    for i in range(samples):
        for j in range(i, (i + RiskMeasureTimeIntervalInDay)):
            # we add the returns over the time interval [i, i + RiskMeasureTimeIntervalInDay]
            added_returns[samples - 1 - i, :] = added_returns[samples - 1 - i, :] + returns[returns.shape[0] - 1 - j, :]
    # linearized loss of the portfolio multiplied by the weights of the WHS
    loss = -portfolioValue * added_returns.dot(weights)
    # we order the losses in decreasing order
    loss_sorted = sorted(loss, reverse=True)
    lambdas_sorted = lambdas
    for i in range(len(loss_sorted)):
        # we order the weights of the WHS following the order of the losses
        lambdas_sorted[i] = lambdas[loss.tolist().index(loss_sorted[i])]
    # we find the greatest i such that sum(lambdas[i:end]) <= 1 - alpha
    i = -1
    lambdas_sum = 0
    while lambdas_sum <= (1 - alpha):
        i += 1
        lambdas_sum += lambdas_sorted[i]
    # Var as the i-th loss
    VaR = float(loss_sorted[i])
    # ES as average of losses greater than the VaR
    ES = float(sum(loss_sorted[0:i] * lambdas_sorted[0:i]) / sum(lambdas_sorted[0:i]))
    return ES, VaR


def PrincCompAnalysis(yearlyCovariance, yearlyMeanReturns, weights, H, alpha, numberOfPrincipalComponents,
                      portfolioValue):
    # spectral decomposition of the variance covariance matrix
    eigenvalues, eigenvectors = linalg.eig(yearlyCovariance)
    gamma = np.zeros((len(eigenvalues), len(eigenvalues)))
    # we order the set of eigenvalues
    eigenvalues_sorted = sorted(eigenvalues, reverse=True)
    weights_sorted = weights
    mean_sorted = yearlyMeanReturns * 0
    for i in range(len(eigenvalues_sorted)):
        # We order the eigenvectors, the weights in the portfolio and the mean vector following the eigenvalues' order
        gamma[:, i] = eigenvectors[:, eigenvalues.tolist().index(eigenvalues_sorted[i])]
        weights_sorted[i] = weights[eigenvalues.tolist().index(eigenvalues_sorted[i])]
        mean_sorted[i] = yearlyMeanReturns[eigenvalues.tolist().index(eigenvalues_sorted[i])]
    # Projected weights
    print(mean_sorted)
    weights_hat = gamma.T.dot(weights_sorted)
    # Projected mean vector
    mean_hat = gamma.T.dot(mean_sorted)
    # reduced standard deviation
    sigma_red = (H * (weights_hat[0: numberOfPrincipalComponents] ** 2).T.dot(eigenvalues_sorted[0: numberOfPrincipalComponents])) ** (1 / 2)
    # reduced mean
    mean_red = H * sum(mean_hat[0: numberOfPrincipalComponents] * weights_hat[0: numberOfPrincipalComponents])
    # VaR and ES with the usual formulas
    VaR = float(portfolioValue * (mean_red + sigma_red * st.norm.ppf(alpha)))
    ES = float(portfolioValue * (mean_red + sigma_red * st.norm.pdf(st.norm.ppf(1 - alpha)) / (1 - alpha)))
    return ES, VaR


def bootstrapStatistical(numberOfSamplesToBootstrap, returns):
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


def BS_PUT(S, K, T, r, d, sigma):
    # B&S formula for a put option
    d1 = (np.log(S / K) + (r - d + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * st.norm.cdf(-d2) - S * np.exp(-d * T) * st.norm.cdf(-d1)


def BS_PUT_delta(S, K, T, r, d, sigma):
    # B&S formula for a put option
    d1 = (np.log(S / K) + (r - d + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    return - np.exp(-d * T) * st.norm.cdf(-d1)


def FullMonteCarloVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,
                      volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears):
    # length of the time interval
    delta = int(math.floor(riskMeasureTimeIntervalInYears * NumberOfDaysPerYears))
    # number of returns we consider, we will add the returns over the corresponding time intervals
    samples = int(len(logReturns) - delta + 1)
    added_returns = np.zeros((samples, 1))
    for i in range(samples):
        for j in range(i, (i + delta)):
            # we add the returns over the time interval [i, i + RiskMeasureTimeIntervalInDay]
            added_returns[samples - 1 - i] = added_returns[samples - 1 - i] + logReturns[len(logReturns) - 1 - j]
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
    loss = - numberOfShares * (simulated_stock - stockPrice * np.ones((len(simulated_stock), 1))) \
           - numberOfPuts * (simulated_put - putPrice * np.ones((len(simulated_put), 1)))
    loss_sorted = sorted(loss, reverse=True)
    # VaR as the 1 - alpha quantile of the loss distribution
    VaR = float(loss_sorted[int(math.floor(samples - 1) * (1 - alpha))])
    return VaR


def DeltaNormalVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,
                   volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears):
    # length of the time interval
    delta = int(math.floor(riskMeasureTimeIntervalInYears * NumberOfDaysPerYears))
    # number of returns we consider, we will add the returns over the corresponding time intervals
    samples = int(len(logReturns) - delta + 1)
    added_returns = np.zeros((samples, 1))
    for i in range(samples):
        for j in range(i, (i + delta)):
            # we add the returns over the time interval [i, i + RiskMeasureTimeIntervalInDay]
            added_returns[samples - 1 - i] = added_returns[samples - 1 - i] + logReturns[len(logReturns) - 1 - j]
    # Time to maturity of the put options minus the delta in years
    sens = BS_PUT_delta(stockPrice, strike, timeToMaturityInYears, rate, dividend, volatility)
    # simulated linearized losses
    loss = - numberOfPuts * stockPrice * sens * added_returns
    loss_sorted = sorted(loss, reverse=True)
    # VaR as the 1 - alpha quantile of the loss distribution
    VaR = float(loss_sorted[int(math.floor(samples - 1) * (1 - alpha))])
    return VaR
