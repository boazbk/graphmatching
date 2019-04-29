import numpy as np
import matplotlib.pyplot as plt

from histogram import histogramPrecision
from subgraph import subgraphPrecision
from subgraph import countsMatching

rs = 10
repeat = 200
n = 100
p = 0.3

# plt.plot(histogramPrecision(n, p, rs, repeat, 3), color="red")
# plt.plot(subgraphPrecision(n, p, rs, repeat, 3), color="green")
plt.plot(countsMatching(n, p, rs, repeat, 5))
plt.show()
