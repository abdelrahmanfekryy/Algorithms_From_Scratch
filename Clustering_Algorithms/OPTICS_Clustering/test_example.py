import numpy as np
import matplotlib.pyplot as plt
from OPTICS_model import OPTICS
from plot_assest import Reachability_Plot,Plot_OPTICS

np.random.seed(14)
C1 = [1, 2] + .4 * np.random.randn(100, 2)
C2 = [0, -1.4] + .8 * np.random.randn(40, 2)
C3 = [3, 0] + .1 * np.random.randn(30, 2)
x = np.vstack((C1, C2, C3))

model = OPTICS(min_samples=10,max_eps=np.inf, xi= 0.2)
model.fit(x)

Plot_OPTICS(model,x)
Reachability_Plot(model)