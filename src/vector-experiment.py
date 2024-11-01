import experts

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

np.random.seed(42)

T = 10 # iterations
c = 5  # dimension
n = 3  # number of experts

y1 = np.rint(np.random.rand(T, c))

E1 = np.random.rand(n, T, c)

A1 = experts.VectorExpertsProblem(E1, y1)

A1.mixture(0.01)

sns.relplot(x = range(10), y = A1.learnerLossVector, kind = "line")

plt.savefig("png/learner-loss.png")

expert_losses = np.sum(A1.expertsLossMatrix, axis = 1)
best_expert_index = np.argmin(expert_losses)

df = pd.DataFrame(
    dict(
        time = np.arange(10),
        learner = A1.learnerLossVector,
        best_expert = A1.expertsLossMatrix[best_expert_index, :]
        )
    )

df_long = pd.melt(df, ['time'])

sns.relplot(x = "time", y = "value", hue = "variable", kind = "line", data = df_long)

plt.savefig("png/best-expert-loss.png")

df_cum = pd.DataFrame(
    dict(
                    time = np.arange(10),
              total_loss = A1.learnerLossVector.cumsum(),
        best_expert_loss = A1.expertsLossMatrix[best_expert_index, :].cumsum()
        )
    )

df_cum_long = pd.melt(df_cum, ['time'])

sns.relplot(x = "time", y = "value", hue = "variable", kind = "line", data = df_cum_long)

plt.savefig("png/cumulative-loss.png")
