#main
import pandas as pd
import numpy as np
import datetime as dt
import scipy as sc
from scipy.stats import norm
import math
import utilities as ut
import control as co

df=pd.read_csv('EUROSTOXX50_2023_Dataset.csv', index_col=0)
(np_num, np_den)=ut.read_our_CSV(df)


#Parameters
alpha=0.95
notional=1e7
delta=1
n_asset=4
weights=np.ones((n_asset,1))/n_asset #we consider a equally weighted portfolio

returns=np.log(np_num/np_den) #computation of the returns
print(np.shape(returns[:,1]))
C=np.corrcoef(returns.T)
print(np.shape(C))
#print(np.shape(returns))

VaR, ES=ut.AnalyticalNormalMeasures(alpha, weights, notional, delta, returns) #we call the implemented function to calculate VaR and ES
print(VaR, ES)
VaR_check=co.plausibilityCheck(returns, weights, alpha, notional, delta)
print(VaR_check)
