import numpy as np
import pandas as pd
import math
import option as opt
from datetime import datetime
from datetime import date





sigma=math.sqrt(0.25)
steps=3
S0=1
delta=1
T=4
tree1=opt.tree_gen(sigma, steps, S0, delta,T)

disc = pd.read_excel('dat_disc.xlsx')
disc.to_numpy()
print(disc)


dates=['2025-02-02','2026-02-02','2027-02-02','2028-02-02']
#sett_date=datetime.strptime(sett_date,"%Y-%m-%d")
#print(sett_date)



sett_date_y=np.zeros((4,1))
for i in range(len(dates)):
    dates[i] = datetime.strptime(dates[i], "%Y-%m-%d")
    sett_date_y[i]=date.toordinal(dates[i])
print(sett_date_y)




