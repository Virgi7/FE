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
disc=disc.to_numpy()
print(disc[:,1])



#Ho messo dentro all'excel solo i discounts che ci interessano e
#con questo comando li leggo e li trasformo in un numpy





