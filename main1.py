# main
import pandas as pd
import numpy as np
import datetime 
from datetime import datetime
import scipy as sc
from scipy.stats import norm
import math
import utilities as ut
import utilities_2 as ut2

df=pd.read_csv('EUROSTOXX50_2023_Dataset.csv', index_col=0)

name_stocks0=['ADSGn.DE', 'ALVG.DE', 'MUVGn.DE',
                            'OREP.PA'] 
dates_num=['2016-03-18','2019-03-20']
dates_den = ['2016-03-17','2019-03-19']

np_num, np_den=ut.read_our_CSV(df,name_stocks0, dates_num, dates_den)

#Parameters
alpha=0.95
notional=1e7
delta=1
n_asset=4
weights=np.ones((n_asset,1))/n_asset #we consider a equally weighted portfolio
returns=np.log(np_num/np_den) #computation of the returns

VaR, ES = ut.AnalyticalNormalMeasures(alpha,weights,notional,delta,returns)
VaR_check=ut.plausibilityCheck(returns, weights, alpha, notional, delta)

#Exercise 1a
#Parameters
sett_date1='2019-03-20'
alpha_1=0.99
dates_num1=dates_num
dates_den1 = dates_den
shares1=np.array([25000, 20000, 20000, 10000])
Nsim=200
name_stocks1a=['TTEF.PA', 'DANO.PA', 'SASY.PA', 'VOWG_p.DE']

stockPrice_1a=df.loc[[sett_date1],name_stocks1a].to_numpy()
ptf_value1a=shares1.dot(stockPrice_1a.T)
weights_1a=(shares1*stockPrice_1a/ptf_value1a).T
np_num1a, np_den1a=ut.read_our_CSV(df,name_stocks1a, dates_num1, dates_den1)
logReturns_1a=np.log(np_num1a/np_den1a)

ES_HSM, VaR_HSM= ut2.HSMeasurements(logReturns_1a, alpha_1, weights_1a, ptf_value1a, delta)
samples_Bootstrap=ut2.bootstrapStatistical(Nsim, logReturns_1a, weights_1a, alpha_1, ptf_value1a,
                         delta)

#Exercise 1b
#Parameters
Nsim=200
name_stocks1b=['ADSGn.DE', 'AIR.PA', 'BBVA.MC', 'BMWG.DE', 'SCHN.PA']
Lambda=0.97
n_asset1b=len(name_stocks1b)
weights_1b=np.ones((n_asset1b,1))/n_asset1b
stockPrice_1b=df.loc[[sett_date1],name_stocks1b].to_numpy()
ptf_value1b=1

np_num1b, np_den1b=ut.read_our_CSV(df,name_stocks1b, dates_num1, dates_den1)
logReturns_1b=np.log(np_num1b/np_den1b)

ES_WHS, VaR_WHS=ut2.WHSMeasurements(logReturns_1b, alpha_1, Lambda, weights_1b, ptf_value1b, delta)

#Exercise 1c
#Parameters
N=20
df_1c=df.loc[:, df.columns != 'ADYEN.AS']
df_1c=df_1c.iloc[: , :N]
name_stocks1c=df_1c.columns
np_num1c, np_den1c=ut.read_our_CSV(df,name_stocks1c, dates_num1, dates_den1)
logReturns_1c=np.log(np_num1c/np_den1c)

n_asset1c=len(name_stocks1c)
weights_1c=np.ones((n_asset1c,1))/n_asset1c
ptf_value1c=1
days_VaR1c=10
n=range(1,7)

ES_PCA=np.zeros((len(n),1))
VaR_PCA=np.zeros((len(n),1))
yearlyCovariance= np.corrcoef(logReturns_1c.T)
yearlyMeanReturns= np.mean(logReturns_1c, axis=0)
for i in n:
    ES_PCA[i], VaR_PCA[i]= ut2.PrincCompAnalysis(yearlyCovariance, yearlyMeanReturns, weights_1c, days_VaR1c, alpha_1, i,
                      ptf_value1c)
    
#Exercise 2
#Parameters
sett_date='2023-01-31'
expiry='2023-04-05'
strike=25
value_ptf=25870000
volatility=0.154
dividend=0.031
alpha_2=0.99
days_VaR=10
rate=0

#We select only the stocks of Vonovia between the settlement date and 2y before
name_stocks2=['VNAn.DE']
n_asset2=len(name_stocks2)
weights_2=np.ones((n_asset2,1))/n_asset2
dates_num=['2021-02-01',sett_date]
dates_den = ['2021-01-29','2023-01-30']
np_num2, np_den2=ut.read_our_CSV(df,name_stocks2, dates_num, dates_den)

stockPrice=np_num2[len(np_num2)-1]
numberOfShares=value_ptf/stockPrice
numberOfPuts=numberOfShares
start = datetime.strptime(sett_date, "%Y-%m-%d")
end = datetime.strptime(expiry, "%Y-%m-%d")
diff = end - start
timeToMaturityInYears=diff.days/365
riskMeasureTimeIntervalInYears=days_VaR/365
NumberOfDaysPerYears=np.busday_count('2022-01-01', '2023-01-01')
logReturns=np.log(np_num2/np_den2)

VaR_MC=ut2.FullMonteCarloVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,                            volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears)
VaR_DN=ut2.DeltaNormalVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,
                            volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears)

#CHECK
VaR_check2=ut.plausibilityCheck(logReturns, weights_2, alpha_2, value_ptf, days_VaR)
