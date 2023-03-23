import numpy as np
import pandas as pd
import math


def tree_gen(sigma, steps, S0, delta, T):#T è la maturity
    u=math.exp(sigma*math.sqrt(delta/steps)) # Delta = 1 anno/n = numero di step per ogni anno
    d=math.exp(-sigma*math.sqrt(delta/steps))
    q=(1 - d) / (u - d)
    tree=np.zeros((int(steps*T/delta+1), int(steps*T/delta+1)))
    tree[0][0]=S0
    for i in range(int(steps*T/delta)):
        for j in range(int(steps*T/delta)):
            tree[i+1][j+1] = tree[i][j]*d
        for j in range(i+1, int(steps*T/delta+1),1):
            tree[i][j] = tree[i][j-1]*u
    return tree[:, range(steps, steps*T+1, steps)],q


def priceCliquet(S0, disc, tree, n, q, rec, SurProbFun, datesInYears):
    # Survival probabilities for the expires in datesInYears (T=0 is included)
    survProb = [SurProbFun(T) for T in datesInYears]
    defProb = [SurProbFun(T-1)-SurProbFun(T) for T in datesInYears]
    T = datesInYears[len(datesInYears)-1]
    payoff = tree
    for i in range(T-1):
        for j in range((i+1)*n):
            for k in range(j, j+n+1):
                payoff[i+1, j] = (tree[i+1, k] - tree[i+1, j] * (i > 0) - S0 * (i == 0)) * q ** ((i+1)*n - k) * (1 - q) ** k
    payoff = payoff * (disc * (survProb + rec * defProb))
    price = sum(sum(payoff))
    return price


