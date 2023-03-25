import numpy as np
import math


def tree_gen(sigma, steps, S0, delta, T):# T è la maturity
    u=math.exp(sigma*math.sqrt(delta/steps)) # Delta = 1 year/n = number of steps for each year
    d=math.exp(-sigma*math.sqrt(delta/steps))
    q=(1 - d) / (u - d)
    tree=np.zeros((int(steps*T/delta+1), int(steps*T/delta+1)))
    tree[0][0]=S0
    for i in range(int(steps*T/delta)):
        for j in range(int(steps*T/delta)):
            tree[i+1][j+1] = tree[i][j]*d
        for j in range(i+1, int(steps*T/delta+1), 1):
            tree[i][j] = tree[i][j-1]*u
    return tree[:, range(steps, steps*T+1, steps)], q


def priceCliquet(S0, disc, tree, n, q, rec, SurProbFun, datesInYears):
    # Survival probabilities for the expires in datesInYears
    survProb = np.array([SurProbFun(T) for T in datesInYears])
    defProb = np.array([(SurProbFun(T-1)-SurProbFun(T))*rec for T in datesInYears])
    T = datesInYears[len(datesInYears)-1]
    payoff = np.zeros(tree.shape)
    for i in range(T):
        if i == 0:
            for j in range(n):
                payoff[j, i] += np.max([(tree[j, i] - S0), 0]) * q ** ((i + 1) * n - j) * (1 - q) ** j
        else:
            for j in range(i*n):
                for k in range(j, j + n + 1):
                    payoff[k, i] += np.max([(tree[k, i] - tree[j, i-1]), 0]) * q ** ((i + 1) * n - k) * (1 - q) ** k
    payoff = payoff * disc * (survProb + defProb)
    price = sum(sum(payoff))
    return price


