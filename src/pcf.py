import numpy as np
import matplotlib.pyplot as plt

import sys
data = np.load("seq_val.npy")
seq_dim = data.shape[1:]
print(data.shape, seq_dim)
data[np.where(data == 0)]=-1
def pbc(i,j):
    d = []
    d.append(np.sqrt(i**2+j**2))
    d.append(np.sqrt((seq_dim[0]-i)**2+j**2))
    d.append(np.sqrt(i**2+(seq_dim[1]-j)**2))
    d.append(np.sqrt((seq_dim[0]-i)**2+(seq_dim[1]-j)**2))
    return min(d)

pcf = {}
err_pcf = {}
for i in range(seq_dim[0]):
    for j in range(seq_dim[1]):
        d = pbc(i,j)
        pcf[d] = np.zeros(data.shape[0])

for i in range(seq_dim[0]):
    for j in range(seq_dim[1]):
        d = pbc(i,j)
        pcf[d] += data[:,i,j]*data[:,0,0]

for k in sorted(list(pcf.keys())):
    err_pcf[k] = np.std(pcf[k])/np.sqrt(pcf[k].shape[0])*1.96
    pcf[k] = np.mean(pcf[k])

plt.plot(sorted(list(pcf.keys())), [pcf[k] for k in sorted(list(pcf.keys()))])
plt.savefig("PCF.png")
pcf_array = np.array([sorted(list(pcf.keys())), [pcf[k] for k in sorted(list(pcf.keys()))], [err_pcf[k] for k in sorted(list(pcf.keys()))]])
np.save("PCF", pcf_array)
