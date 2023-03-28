import statistics

import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import scipy as sc
import scipy.stats as stat
import math
import utilities as ut
import utilities_2 as ut2
import option as opt
from scipy.interpolate import interpolate

df=pd.read_csv('EUROSTOXX50_2023_Dataset.csv', index_col=0)
#df = df.fillna(method='ffill')
name_stocks0=['ADSGn.DE', 'ALVG.DE', 'MUVGn.DE', 'OREP.PA']
dates_num=['2016-03-18','2019-03-20']
dates_den = ['2016-03-17','2019-03-19']

np_num, np_den=ut.read_our_CSV(df,name_stocks0, dates_num, dates_den)
alpha=0.95
notional=1e7
delta=1
n_asset=4
weights=np.ones((n_asset,1))/n_asset
returns=np.log(np_num/np_den)
VaR, ES=ut.AnalyticalNormalMeasures(alpha, weights, notional, delta, returns)
print("es 0", VaR, ES)
#CHECK
VaR_check=ut.plausibilityCheck(returns, weights, alpha, notional, delta)
print("check 0", VaR_check)

# EXERCISE 1
name_stocks0=['TTEF.PA', 'DANO.PA', 'SASY.PA', 'VOWG_p.DE']
dates_num=['2016-03-18','2019-03-20']
dates_den = ['2016-03-17','2019-03-19']

np_num, np_den=ut.read_our_CSV(df,name_stocks0, dates_num, dates_den)
alpha = 0.99
shares = [25000, 20000, 20000, 10000]
k = np_num.shape[0] - 1
notional = sum(shares*np_num[k, :])
delta = 1
weights = shares*np_num[k, :] / notional
returns = np.log(np_num/np_den)
ES, VaR = ut2.HSMeasurements(returns, alpha, weights, notional, delta)
print("hs", VaR, ES)
#CHECK
VaR_check=ut.plausibilityCheck(returns, weights, alpha, notional, delta)
print("check hs",VaR_check)
samples = ut2.bootstrapStatistical(200, returns)
VaR_boot = ut2.HSMeasurements(samples, alpha, weights, notional, delta)
print("boot", np.mean(VaR_boot))

# EXERCISE 1
name_stocks0=['ADSGn.DE', 'AIR.PA', 'BBVA.MC', 'BMWG.DE', 'SCHN.PA']
dates_num=['2016-03-18','2019-03-20']
dates_den = ['2016-03-17','2019-03-19']

np_num, np_den = ut.read_our_CSV(df,name_stocks0, dates_num, dates_den)
alpha = 0.99
weights = 0.2 * np.ones((len(name_stocks0),1))
notional = 10000000
delta = 1
returns = np.log(np_num/np_den)
ES, VaR = ut2.WHSMeasurements(returns, alpha, 0.97, weights, notional, delta)
print("whs", VaR, ES)
#CHECK
VaR_check = ut.plausibilityCheck(returns, weights, alpha, notional, delta)
print("whs", VaR_check)

# EXERCISE 1
name_stocks0=['ABI.BR', 'AD.AS', 'ADSGn.DE', 'AIR.PA', 'AIRP.PA', 'ALVG.DE', 'ASML.AS', 'AXAF.PA', 'BASFn.DE',
              'BAYGn.DE', 'BBVA.MC', 'BMWG.DE', 'BNPP.PA', 'CRH.I', 'DANO.PA', 'DB1Gn.DE', 'DPWGn.DE', 'DTEGn.DE',
              'ENEI.MI', 'ENI.MI']
dates_num=['2016-03-18','2019-03-20']
dates_den = ['2016-03-17','2019-03-19']

np_num, np_den=ut.read_our_CSV(df,name_stocks0, dates_num, dates_den)
alpha = 0.99
weights = 0.05 * np.ones((len(name_stocks0),1))
notional = 100000000
delta = 10
returns = np.log(np_num/np_den)
yearlyCovariance = np.cov(returns.T)
yearlyMeanReturns = np.zeros((20, 1))
for i in range(20):
    yearlyMeanReturns[i] = np.mean(returns[:, i])
ES, VaR = ut2.PrincCompAnalysis(yearlyCovariance, yearlyMeanReturns, weights, delta, 0.99, 6, notional)
print("pca", VaR, ES)
#CHECK
VaR_check = ut.plausibilityCheck(returns, weights, alpha, notional, delta)
print("check pca", VaR_check)

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
dates_num=['2021-02-01',sett_date]
dates_den = ['2021-01-29','2023-01-30']

np_num2, np_den2=ut.read_our_CSV(df,name_stocks2, dates_num, dates_den)

stockPrice=np_num2[len(np_num2)-1]
numberOfShares = value_ptf / stockPrice
numberOfPuts = numberOfShares
start = datetime.strptime(sett_date, "%Y-%m-%d")
end = datetime.strptime(expiry, "%Y-%m-%d")
diff = end - start
timeToMaturityInYears=diff.days/365
riskMeasureTimeIntervalInYears=days_VaR/365
NumberOfDaysPerYears=260
logReturns=np.log(np_num2/np_den2)
VaR_MC=ut2.FullMonteCarloVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,
                            volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears)
VaR_DN=ut2.DeltaNormalVaR(logReturns, numberOfShares, numberOfPuts, stockPrice, strike, rate, dividend,
                            volatility, timeToMaturityInYears, riskMeasureTimeIntervalInYears, alpha, NumberOfDaysPerYears)
print("mc", VaR_MC, VaR_DN)


#EXERCISE 3
sigma = 0.25
steps = 34
S0 = 2.41
delta = 1
T = 4
rec = 0.4
tree= opt.tree_gen(sigma, steps, S0, delta, T)
df = pd.read_excel('dat_disc.xlsx')
df=df.to_numpy()
priceBS = opt.priceCliquetBS(S0, df[:,2], tree, steps, sigma, rec, df[:,3], df[1:,1])
print(priceBS)