import numpy as np
from numpy import linalg
import math
import scipy.stats as st
import random


def HSMeasurements(returns, alpha, weights, portfolioValue, RiskMeasureTimeIntervalInDay):
    nsamples = int(returns.shape[0] / RiskMeasureTimeIntervalInDay)
    addedreturns = np.zeros((nsamples, returns.shape[1]))
    for i in range(nsamples):
        for j in range(i * RiskMeasureTimeIntervalInDay, ((i + 1) * RiskMeasureTimeIntervalInDay)):
            addedreturns[i, :] = addedreturns[i, :] + returns[j, :]
    loss = -portfolioValue * addedreturns.dot(weights)
    loss_sorted = sorted(loss)
    VaR = loss_sorted[math.floor(nsamples * (1 - alpha)) - 1]
    ES = np.mean(loss_sorted[0:math.floor(nsamples * (1 - alpha)) - 1])
    return [VaR, ES]


def WHSMeasurements(returns, alpha, llambda, weights, portfolioValue, RiskMeasureTimeIntervalInDay):
    nsamples = int(returns.shape[0] / RiskMeasureTimeIntervalInDay)
    C = (1 - llambda) / (1 - llambda ** nsamples)
    lambdas = np.zeros((nsamples, 1))
    for i in range(nsamples):
        lambdas[i] = C * llambda ** (returns.shape[0] - returns.shape[0] * i / nsamples)
    addedreturns = np.zeros((nsamples, returns.shape[1]))
    for i in range(nsamples):
        for j in range(i * RiskMeasureTimeIntervalInDay, ((i + 1) * RiskMeasureTimeIntervalInDay)):
            addedreturns[i, :] = addedreturns[i, :] + returns[j, :]
    loss = -portfolioValue * addedreturns.dot(weights)
    loss_sorted = sorted(loss)
    lambdas_sorted = lambdas
    for i in range(len(loss_sorted)):
        lambdas_sorted[i] = lambdas[loss == loss_sorted[i]]
    i = 0
    lambdas_sum = 0
    while lambdas_sum <= (1 - alpha):
        lambdas_sum += lambdas_sorted[i]
        i += 1
    i += -1
    print(i)
    VaR = loss_sorted[i]
    ES = (sum(loss_sorted[1:i] * lambdas_sorted[1:i]) + loss_sorted[0] * lambdas_sorted[0]) / (
                sum(lambdas_sorted[1:i]) + lambdas[0])
    return [ES, VaR]


def PrincCompAnalysis(yearlyCovariance, yearlyMeanReturns, weights, H, alpha, numberOfPrincipalComponents,
                      portfolioValue):
    eigenvalues, eigenvectors = linalg.eigvals(yearlyCovariance)
    gamma = np.zeros((len(eigenvalues), len(eigenvalues)))
    eigenvalues_sorted = sorted(eigenvalues)
    weights_sorted = weights
    mean_sorted = yearlyMeanReturns
    for i in range(len(eigenvalues_sorted)):
        gamma[:, i] = eigenvectors[eigenvalues == eigenvalues_sorted[i]]
        weights_sorted[:, i] = weights[eigenvalues == eigenvalues_sorted[i]]
        mean_sorted[:, i] = yearlyMeanReturns[eigenvalues == eigenvalues_sorted[i]]
    weights_hat = gamma.T.dot(weights_sorted)
    mean_hat = gamma.T.dot(mean_sorted)
    sigma_red = (H * sum(
        eigenvalues_sorted[0:numberOfPrincipalComponents] * weights_hat[0:numberOfPrincipalComponents] ** 2)) ** (1 / 2)
    mean_red = H * sum(mean_hat[0:numberOfPrincipalComponents] * weights_hat[0:numberOfPrincipalComponents] ** 2)
    VaR = portfolioValue * (mean_red + sigma_red * st.norm.ppf(alpha))
    ES = portfolioValue * (mean_red + sigma_red * st.norm.pdf(st.norm.ppf(alpha)) / (1 - alpha))
    return [ES, VaR]


def bootstrapStatistical(numberOfSamplesToBootstrap, returns, alpha, portfolioValue, RiskMeasureTimeIntervalInDay):
    n = returns.shape[1]
    samples = np.zeros(numberOfSamplesToBootstrap, 1)
    for i in range(numberOfSamplesToBootstrap):
        x = int(random.randint(0, n - 1))
        samples[i] = HSMeasurements(returns[:, x], alpha, 1, portfolioValue, RiskMeasureTimeIntervalInDay)
    return samples
