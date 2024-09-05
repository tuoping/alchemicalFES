import numpy as np
import sys
seq_dim=(6,6)

def ReadReferenceF(filename):
    ofile_prob_E = open(filename,"r")
    Reference_dict = {}
    idx_jj = 0
    while(True):
        line = ofile_prob_E.readline()
        if not line:
            break

        line = ofile_prob_E.readline()
        bin_centers = np.array([float(x) for x in line.split()])

        line = ofile_prob_E.readline()
        jj = float(line.split()[-1].replace("kBT=",""))

        line = ofile_prob_E.readline()
        F = np.array([float(x) for x in line.split()])

        Reference_dict[jj]=np.stack([bin_centers, F])
    ofile_prob_E.close()
    return Reference_dict

Tpred = float(sys.argv[1])
Tref = float(sys.argv[2])

import os
fp_Tpred = open("F-MAGN-kBT3.20-T%.2f.dat"%Tpred, "r")
fp_Tpred.readline()
bincenters_Tpred = [float(x) for x in fp_Tpred.readline().split()]
fp_Tpred.readline()
F_Tpred = [float(x) for x in fp_Tpred.readline().split()]

ref_dirname = "/nfs/scistore14/chenggrp/ptuo/NeuralRG/dirichlet-flow-matching-test3/data/ising-latt%dx%d-T4.0/latt%dx%d/"%(*seq_dim, *seq_dim)
Reference_dict = ReadReferenceF(os.path.join(ref_dirname, "F-MAGN-REF.dat"))
bincenters_Tref = Reference_dict[Tref][0]
F_Tref = Reference_dict[Tref][1]

import matplotlib.pyplot as plt
plt.scatter(bincenters_Tpred, F_Tpred, label="Predicted FES of diffusion temperature %.2f"%Tpred)
plt.plot(bincenters_Tref, F_Tref, label="Reference FES of temperature %.2f"%Tref)
plt.legend(fontsize=14)
plt.xlabel("Magnetization", fontdict={"size":14})
plt.ylabel("Negative likelihood ($k_BT$)", fontdict={"size":14})
plt.savefig("path-T%.2f-refT%.2f.png"%(Tpred, Tref), bbox_inches="tight")

