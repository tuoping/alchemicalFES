import numpy as np
import sys
temp = float(sys.argv[1])
s = np.loadtxt("S%.2f.txt"%temp).reshape(-1,36*36)
np.save("buffer-S%.2f.npy"%temp,s)
# s = np.load("buffer-S%.2f.npy"%temp)

from collections import Counter
magn = np.arange(-36*36, 36*36+1, 2).reshape(-1,1)

import matplotlib.pyplot as plt
data = np.sum(s, axis=-1)
counter = Counter(data)
histgram = np.array([counter[m] for m in magn[:,0]])
P = histgram/np.sum(histgram)
free_energies = -np.log(P)*temp
plt.figure()
plt.plot(magn[:,0], free_energies)
with open(f"FES%.2f.dat"%temp, "w") as fp:
    for i in range(len(magn)):
        fp.write("%.2f  %.4f\n"%(magn[i,0], free_energies[i]))
plt.savefig("F-S%.2f.png"%temp)
plt.figure()
plt.plot(magn[:,0], histgram)
plt.semilogy()
plt.savefig("H-S%.2f.png"%temp)
