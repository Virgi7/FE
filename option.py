import numpy as np
import pandas as pd
import math


def tree_gen(sigma, steps, S0, delta,T):#T è la maturity
    u=math.exp(sigma*math.sqrt(delta/steps)) #Delta=1 anno/n=numero di step per ogni anno
    d=math.exp(-sigma*math.sqrt(delta/steps))
    tree=np.zeros((int(steps*T/delta+1),int(steps*T/delta+1)))
    tree[0][0]=S0
    for i in range (int(steps*T/delta)):
        for j in range (int(steps*T/delta)):
            tree[i+1][j+1]=tree[i][j]*d
        for j in range(i+1,int(steps*T/delta+1),1):
            tree[i][j]=tree[i][j-1]*u
    return tree[:,range(steps,steps*T+1,steps)]





