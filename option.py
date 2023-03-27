import numpy as np
import math
import scipy.stats as st


def tree_gen(sigma, steps, S0, delta, T):# T è la maturity
    u = math.exp(sigma*math.sqrt(delta/steps))# Delta = 1 year/n = number of steps for each year
    d = math.exp(-sigma*math.sqrt(delta/steps))
    q = (1 - d) / (u - d)
    tree = np.zeros((int(steps*T/delta + 1), int(steps*T/delta + 1)))
    tree[0][0] = S0
    for i in range(1, int(steps*T/delta) + 1):
        for j in range(i + 1):
            tree[j][i] = S0 * (u ** (i - j)) * (d ** j)
    return tree[:, range(steps, steps * T + 1, steps)]


def priceCliquet(S0, disc, tree, n, rec, sigma, SurProbFun, datesInYears):
    u = math.exp(sigma * math.sqrt(1 / n))
    d = math.exp(-sigma * math.sqrt(1 / n))
    q = (1 - d)/(u - d)
    survProb = np.array([SurProbFun(T) for T in datesInYears])
    defProb = np.array([(SurProbFun(T-1)-SurProbFun(T))*rec for T in datesInYears])
    T = len(datesInYears)
    payoff = np.zeros(tree.shape)
    for i in range(T):
        if i == 0:
            for j in range(n + 1):
                payoff[j, i] += (tree[j, i] - S0) * float((tree[j, i] > S0)) * bincoeff(n, j) * (q ** (n - j)) * ((1 - q) ** j)
        else:
            for j in range(i * n + 1):
                for k in range(j, j + n + 1):
                    payoff[k, i] += (tree[k, i] - tree[j, i - 1]) * float((tree[k, i] > tree[j, i - 1])) * (bincoeff(n, n - k + j) * bincoeff(i * n, i * n - j)) * (q ** (n - k + j)) * ((1 - q) ** (k - j)) * (q ** (i * n - j)) * ((1 - q) ** j)
    price = payoff * disc[0: len(disc) - 1] * (survProb + defProb)
    price = sum(sum(price))
    return price


def priceCliquetBS(S0, disc, tree, n, sigma, rec, SurProbFun, datesInYears):
    u = math.exp(sigma * math.sqrt(1 / n))
    d = math.exp(-sigma * math.sqrt(1 / n))
    q = (1 - d)/(u - d)
    # Survival probabilities for the expires in datesInYears
    survProb = np.array([SurProbFun(T) for T in datesInYears])
    defProb = np.array([(SurProbFun(T-1)-SurProbFun(T))*rec for T in datesInYears])
    T = len(datesInYears)
    payoff = np.zeros(tree.shape)
    payoff[0, 0] = BS_CALL(S0, S0, datesInYears[0], - np.log(disc[1])/datesInYears[1], 0, sigma)
    for i in range(1, T):
        for j in range(i * n + 1):
            TTM = datesInYears[i] - datesInYears[i - 1]
            payoff[j, i] = BS_CALL(tree[j, i-1], tree[j, i-1], TTM, - np.log(disc[i + 1])/datesInYears[i], 0, sigma) * bincoeff(i * n, j) * (q ** (i * n - j)) * ((1 - q) ** j)
    price = payoff * disc[0: len(disc) - 1] * (survProb + defProb)
    price = sum(sum(price))
    return price


def bincoeff(n, k):
    if k == 0 or k == n:
        coeff = 1
    else:
        if k > n:
            coeff = 0
        else:
            a = math.factorial(n)
            b = math.factorial(k)
            c = math.factorial(n - k)
            coeff = a / (b * c)
    return coeff


def BS_CALL(S, K, T, r, d, sigma):
    # B&S formula for a put option
    d1 = (np.log(S / K) + (r - d + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-d * T) * st.norm.cdf(d1) - K * np.exp(-r * T) * st.norm.cdf(d2)
