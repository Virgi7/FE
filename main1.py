#main
import pandas as pd
import numpy as np
import datetime as dt
import scipy as sc
from scipy.stats import norm
import math
import utilities as ut

df=pd.read_csv('EUROSTOXX50_2023_Dataset.csv', index_col=0)
df=df.fillna(method='ffill') #we fill the missing values of the stocks with the previous known ones

#We select stocks properly in order to perform the right computation of the returns
df_ptf=df[824:1592] #we selected 3y from 20-03-2019 bckw. up to 22-03-2016
df_ptf=df_ptf.loc[:,['ADSGn.DE','ALVG.DE','MUVGn.DE','OREP.PA']] #we select only the 4 columns corresponding to Adidas, Allianz, Munich RE and l'Oreal
df_den=df[825:1593] #we selected 3y from one day before 20-03-2019 (19-03-2019) up one day before the last date of df_ptf (21-03-2016)
df_den=df_den.loc[:,['ADSGn.DE','ALVG.DE','MUVGn.DE','OREP.PA']] 

#we pass to numpy arrays to perform the logarithm
np_den=df_den.to_numpy() 
np_num=df_ptf.to_numpy()

#Parameters
alpha=0.95
notional=1e7
delta=1
n_asset=4
weights=np.ones((n_asset,1))/n_asset #we consider a equally weighted portfolio

returns=np.log(np_num/np_den) #computation of the returns

VaR, ES=ut.AnalyticalNormalMeasures(alpha, weights, notional, delta, returns) #we call the implemented function to calculate VaR and ES
print(VaR, ES)
