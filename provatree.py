import numpy as np
import pandas as pd
import math
import option as opt
from datetime import datetime
from datetime import date

sigma = 0.25
steps = 34
S0 = 2.41
delta = 1
T = 4
rec = 0.4
tree= opt.tree_gen(sigma, steps, S0, delta, T)


df = pd.read_excel('dat_disc.xlsx')
df=df.to_numpy()
SurvProbs=df[:,3]
priceBS = opt.priceCliquetBS(S0, df[:,2], tree, steps, sigma, rec, SurvProbs, df[1:,1])
print(priceBS)







