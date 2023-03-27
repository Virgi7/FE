import numpy as np
import pandas as pd
import math
import option as opt

sigma = 0.25
steps = 34
S0 = 2.41
delta = 1
T = 4
rec = 0.4
tree, q = opt.tree_gen(sigma, steps, S0, delta, T)


def SurProbFun(t):
    intensities = [0.00500001041670575, 0.00635686394536413, 0.00722384773356583, 0.00756985690474914, 0.00739211576159149, 0.00692374267316075, 0.00640219978998205]
    prob = 1
    for i in range(len(intensities)):
        prob = prob * np.exp(- np.max([np.min([t - i, 1]), 0]) * intensities[i])
    return prob


price = opt.priceCliquet(S0, [0.968072448, 0.93889066, 0.914635392, 0.89163513], tree, steps, q, 0.4, SurProbFun, [1, 2.0054794521, 3.0054794521, 4.0054794521])
print(price)
priceBS = opt.priceCliquetBS(S0, [1, 0.968072448, 0.93889066, 0.914635392, 0.89163513], tree, steps, q, sigma, 0.4, SurProbFun, [1, 2.0054794521, 3.0054794521, 4.0054794521])
print(priceBS)
print(50 * (priceBS - price) * 10 ** 6)

