import numpy as np
import pandas as pd
import math
import option as opt




sigma=math.sqrt(0.25)
steps=3
S0=1
delta=1
T=4
tree1=opt.tree_gen(sigma, steps, S0, delta,T)
print(tree1)
