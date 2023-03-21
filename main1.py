# main
import pandas as pd
import numpy as np
import utilities as ut
<<<<<<< HEAD
import control as co
=======
import utilities_2 as ut2
>>>>>>> f409da27cd1282f822c34db34f496db521f6831b

df = pd.read_csv('EUROSTOXX50_2023_Dataset.csv', index_col=0)
(np_num, np_den) = ut.read_our_CSV(df)

<<<<<<< HEAD

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
=======
# Parameters
alpha = 0.95
notional = 1e7
delta = 3
n_asset = 4
weights = np.ones((n_asset, 1))/n_asset  # we consider a equally weighted portfolio

returns = np.log(np_num/np_den)  # computation of the returns
>>>>>>> f409da27cd1282f822c34db34f496db521f6831b

# we call the implemented function to calculate VaR and ES
VaR, ES = ut2.HSMeasurements(returns, alpha, weights, notional, delta)
print(VaR, ES)
VaR_check=co.plausibilityCheck(returns, weights, alpha, notional, delta)
print(VaR_check)
