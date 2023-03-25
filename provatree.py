import numpy as np
import pandas as pd
import math
import option as opt

sigma = math.sqrt(0.25)
steps = 24
S0 = 1
delta = 1
T = 4
tree, q = opt.tree_gen(sigma, steps, S0, delta, T)


def SurProbFun(t):
    return np.exp(4 * t * 10 ** (-4))


price = opt.priceCliquet(S0, [0.94, 0.9, 0.88, 0.87], tree, steps * T, q, 0.3, SurProbFun, [1, 2, 3, 4])
