import numpy as np
from numpy import linalg
import utilities_2 as ut2
import math
returns = np.ones((10, 4))
for i in range(10):
    returns[i, :] = returns[i, :] * (-0.55**i) + (-0.7)**i+2 + math.exp(-0.05*i-10) - (i>0)*returns[i-1, :]
alpha = 0.99
weights = [0.25, 0.25, 0.25, 0.25]
portfolioValue = 5
RiskMeasureTimeIntervalInDay = 1
lam = 0.97
[VaR, ES] = ut2.WHSMeasurements(returns, alpha, lam, weights, portfolioValue, RiskMeasureTimeIntervalInDay)
print(VaR)
print(ES)
D=np.zeros((4,4))
D[0,0]=3
D[1,1]=1
D[2,2]=4
D[3,3]=2
print(D)
print(linalg.eigvals(D))
a=linalg.eig(D)
print(linalg.eig(D))
print(a)
