# Functions used to price a cliquet option
import numpy as np
import math
import scipy.stats as st
# tree_gen(sigma, steps, S0, delta, T) generates the binomial tree for the given underlying asset
# priceCliquetBS(S0, disc, tree, n, sigma, rec, SurProb, datesInYears) prices the cliquet option
# bincoeff(n, k) computes the binomial coefficient
# BS_PUT(S, K, T, r, d, sigma) B&S price for Put option
# BS_PUT_delta(S, K, T, r, d, sigma) B&S delta for Put option
# BS_CALL(S, K, T, r, d, sigma) B&S price for Call option


def tree_gen(sigma, steps, S0, delta, T):  # T è la maturity
    u = math.exp(sigma*math.sqrt(delta/steps))  # Delta = 1 year/n = number of steps for each year
    d = math.exp(-sigma*math.sqrt(delta/steps))
    tree = np.zeros((int(steps*T/delta + 1), int(steps*T/delta + 1)))
    tree[0][0] = S0
    for i in range(1, int(steps*T/delta) + 1):
        for j in range(i + 1):
            tree[j][i] = S0 * (u ** (i - j)) * (d ** j)
    return tree[:, range(steps, steps * T + 1, steps)]


def priceCliquetBS(S0, disc, tree, n, sigma, rec, SurProb, datesInYears):
    # up and down in the tree
    u = math.exp(sigma * math.sqrt(1 / n))
    d = math.exp(-sigma * math.sqrt(1 / n))
    # probability of up
    q = (1 - d)/(u - d)
    # Survival probabilities for the expires in datesInYears
    DefProbRec = (SurProb[0: len(SurProb) - 1] - SurProb[1: len(SurProb)]) * rec
    T = len(datesInYears)
    payoff = np.zeros(tree.shape)
    # we consider the payments as payoff of ATM call options, the premium computed with B&S formula
    payoff[0, 0] = BS_CALL(S0, S0, datesInYears[0], - np.log(disc[1])/datesInYears[0], 0, sigma)
    for i in range(1, T):
        for j in range(i * n + 1):
            TTM = datesInYears[i] - datesInYears[i - 1]
            payoff[j, i] = BS_CALL(tree[j, i-1], tree[j, i-1], TTM, - np.log(disc[i + 1]/disc[i])/TTM, 0, sigma) * bincoeff(i * n, j) * (q ** (i * n - j)) * ((1 - q) ** j)
    # We multiply by the discounts, the survival probabilities and the recovery multiplied by the default probability in each time interval
    price = payoff * disc[0: len(disc) - 1] * (SurProb[1: len(SurProb)] + DefProbRec)
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


def BS_PUT(S, K, T, r, d, sigma):
    # B&S formula for a put option
    d1 = (np.log(S / K) + (r - d + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * st.norm.cdf(-d2) - S * np.exp(-d * T) * st.norm.cdf(-d1)


def BS_PUT_delta(S, K, T, r, d, sigma):
    # B&S formula for a put option
    d1 = (np.log(S / K) + (r - d + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    return - np.exp(-d * T) * st.norm.cdf(-d1)


def BS_CALL(S, K, T, r, d, sigma):
    # B&S formula for a put option
    d1 = (np.log(S / K) + (r - d + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-d * T) * st.norm.cdf(d1) - K * np.exp(-r * T) * st.norm.cdf(d2)
