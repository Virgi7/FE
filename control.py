import math

import pandas as pd
import utilities as ut
import numpy as np
import scipy
from scipy.stats import norm


def plausibilityCheck(returns, portfolioWeights, alpha, portfolioValue, riskMeasureTimeIntervalInDay):
    l = np.zeros((len(portfolioWeights), 1))
    u = np.zeros((len(portfolioWeights), 1))
    sens = np.zeros((len(portfolioWeights), 1))
    av = np.zeros((len(portfolioWeights), 1))
    sVaR = np.zeros((len(portfolioWeights), 1))
    for i in range (len(portfolioWeights)):
        l[i]=np.quantile(returns[:,i],1-alpha)
        u[i]=np.quantile(returns[:,i],alpha)
        sens[i] = portfolioValue * portfolioWeights[i]
        av[i]=(abs(l[i])+abs(u[i]))/2
        sVaR[i]=sens[i]*av[i]
    print(sVaR)
    C=np.corrcoef(returns.T)
    VaR=math.sqrt((sVaR.T).dot(C.dot(sVaR)))
    return VaR
