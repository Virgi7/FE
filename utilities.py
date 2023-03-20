#Funzioni Assigment 5.0
import pandas as pd
import numpy as np
import scipy
import math
from scipy.stats import norm



def AnalyticalNormalMeasures(alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay,returns):
    Loss=-portfolioValue*(returns.dot(weights)) #moltiplico per i pesi perchè returns è una matrice num_datexnum_assets e weights è n_assetx1
    Loss_mean=np.mean(Loss)
    Loss_std=np.std(Loss)
    VaR_std=norm.ppf(alpha)
    VaR=riskMeasureTimeIntervalInDay*Loss_mean+math.sqrt(riskMeasureTimeIntervalInDay)*Loss_std*VaR_std
    ES_std=norm.pdf(VaR_std)/(1-alpha)
    ES=riskMeasureTimeIntervalInDay*Loss_mean+math.sqrt(riskMeasureTimeIntervalInDay)*Loss_std*ES_std
    return VaR, ES



