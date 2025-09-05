import numpy as np
import matplotlib.pyplot as plt

tovar = np.load("ELV_ld_N10_D_setloadtest.npy")
datavar = np.var(tovar,axis=0)
dataR = np.load("ELV_ld_N10_D_setloadtest_Rs.npy")

plt.plot(dataR, datavar)

plt.xscale('log')
plt.yscale('log')

plt.show()
