import numpy as np
import matplotlib.pyplot as plt
elst = np.loadtxt('result')
efci = -7.939344693518
plt.axhline(y=efci,lw=2)
plt.plot(elst,'ro-')
plt.savefig('convergence.pdf')
plt.show()
