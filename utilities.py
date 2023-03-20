#Funzioni Assigment 5.0
import pandas as pd
import numpy as np
import scipy
import math
from scipy.stats import norm



def AnalyticalNormalMeasures(alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay,returns):
    #We calculate the loss as the portfolio's value in t multiplied by the scalar product between wt and xt
    Loss=-portfolioValue*(returns.dot(weights)) #np_array.dot computes the product between matrices
    Loss_mean=np.mean(Loss) #We calculate the mean of losses
    Loss_std=np.std(Loss) #We calculate the standard deviation of losses
    VaR_std=norm.ppf(alpha) #VaR of a std normal is the inverse of the cdf evaluated in alpha (N^(-1)(alpha))
    VaR=riskMeasureTimeIntervalInDay*Loss_mean+math.sqrt(riskMeasureTimeIntervalInDay)*Loss_std*VaR_std #VaR expressed in function of VaR_std with delta=current time window (1 in this case)
    ES_std=norm.pdf(VaR_std)/(1-alpha) #ES for a std normal is the pdf of a normal evaluated in the VaR of a std normal
    ES=riskMeasureTimeIntervalInDay*Loss_mean+math.sqrt(riskMeasureTimeIntervalInDay)*Loss_std*ES_std #ES expressed in function of ES_std
    return VaR, ES



