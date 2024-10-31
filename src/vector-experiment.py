import experts

import numpy as np
import pandas as pd

np.random.seed(42)

T = 10 # iterations
c = 5  # dimension
n = 3  # number of experts

y1 = np.rint(np.random.rand(T, c))

E1 = np.random.rand(n, T, c)

A1 = experts.VectorExpertsProblem(E1, y1)

A1.mixture(0.01)

print(A1.learnerLossVector)
