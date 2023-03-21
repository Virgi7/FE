# main
import pandas as pd
import numpy as np
import utilities as ut
import utilities_2 as ut2

df = pd.read_csv('EUROSTOXX50_2023_Dataset.csv', index_col=0)
(np_num, np_den) = ut.read_our_CSV(df)

# Parameters
alpha = 0.95
notional = 1e7
delta = 3
n_asset = 4
weights = np.ones((n_asset, 1))/n_asset  # we consider a equally weighted portfolio

returns = np.log(np_num/np_den)  # computation of the returns

# we call the implemented function to calculate VaR and ES
VaR = ut2.DeltaNormalVaR(returns[:, 0], 2500, 2500, 26.8, 25, 0, 0.031, 0.154, 1/3, 1/260, alpha, 260)
print(VaR)
