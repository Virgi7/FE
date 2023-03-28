# Functions Assignment 5.0
import numpy as np
from scipy.stats import norm
import utilities_2 as ut2
from numpy import linalg
import math
import scipy.stats as st
import random
import option as opt


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
    added_returns = ut2.aggregateReturns(returns, riskMeasureTimeIntervalInDay)
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
    added_returns = ut2.aggregateReturns(returns, RiskMeasureTimeIntervalInDay)
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
    added_returns = ut2.aggregateReturns(returns, RiskMeasureTimeIntervalInDay)
    # weights of the Historical Simulation
    lambdas = ut2.WHSweights(Lambda, added_returns.shape[0])
    # linearized loss of the portfolio multiplied by the weights of the WHS
    loss = -portfolioValue * added_returns.dot(weights)
    # we order the losses in decreasing order
    loss_sorted = sorted(loss, reverse=True)
    # We sort the weights in a way that they correspond to the ordered losses
    lambdas_sorted = ut2.sort_as(loss, loss_sorted, lambdas)
    # we find the greatest i such that sum(lambdas[i:end]) <= 1 - alpha
    i = ut2.searchLevel(lambdas_sorted, alpha)
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
    eigenvalues_sorted = sorted(eigenvalues, reverse=True)
    weights_sorted = ut2.sort_as(eigenvalues, eigenvalues_sorted, weights)
    mean_sorted = ut2.sort_as(eigenvalues, eigenvalues_sorted, yearlyMeanReturns)
    gamma = np.zeros((len(eigenvalues), len(eigenvalues)))
    for i in range(len(eigenvalues_sorted)):
        # We order the eigenvectors, the weights in the portfolio and the mean vector following the eigenvalues' order
        gamma[:, i] = ut2.sort_as(eigenvalues, eigenvalues_sorted, eigenvectors[:, i])
    # Projected weights
    weights_hat = gamma.T.dot(weights_sorted)
    # Projected mean vector
    mean_hat = gamma.T.dot(mean_sorted)
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
    added_returns = ut2.aggregateReturns(logReturns, delta)
    # simulated stock price
    simulated_stock = stockPrice * np.exp(added_returns)
    # Time to maturity of the put options minus the delta in years
    TTM_simulated = timeToMaturityInYears - riskMeasureTimeIntervalInYears
    simulated_put = np.zeros((len(simulated_stock), 1))
    for i in range(len(simulated_stock)):
        # B&S formula applied to the simulated stock price
        simulated_put[i] = opt.BS_PUT(simulated_stock[i], strike, TTM_simulated, rate, dividend, volatility)
    # price today of the put option
    putPrice = opt.BS_PUT(stockPrice, strike, timeToMaturityInYears, rate, dividend, volatility)
    # simulated losses
    loss = - numberOfShares * (simulated_stock - stockPrice * np.ones((len(simulated_stock), 1))) - numberOfPuts * (simulated_put - putPrice * np.ones((len(simulated_put), 1)))
    loss_sorted = sorted(loss, reverse=True)
    # VaR as the 1 - alpha quantile of the loss distribution
    VaR = np.quantile(loss_sorted, alpha)
    return VaR


def DeltaNormalVaR(logReturns, numberOfPuts, stockPrice, strike, rate, dividend,
                   volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears):
    # length of the time interval
    delta = int(math.floor(riskMeasureTimeIntervalInYears * NumberOfDaysPerYears))
    # number of returns we consider, we will add the returns over the corresponding time intervals
    added_returns = ut2.aggregateReturns(logReturns, delta)
    # Time to maturity of the put options minus the delta in years
    sens = opt.BS_PUT_delta(stockPrice, strike, timeToMaturityInYears, rate, dividend, volatility)
    # simulated linearized losses
    loss = - numberOfPuts * stockPrice * sens * added_returns
    loss_sorted = sorted(loss, reverse=True)
    # VaR as the 1 - alpha quantile of the loss distribution
    VaR = np.quantile(loss_sorted, alpha)
    return VaR
