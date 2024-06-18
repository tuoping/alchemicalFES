import numpy as np
import matplotlib.pyplot as plt
def ReadReferenceF(filename):
    ofile_prob_E = open(filename,"r")
    line = ofile_prob_E.readline()
    line = ofile_prob_E.readline()
    bin_centers = np.array([float(x) for x in line.split()])

    line = ofile_prob_E.readline()
    line = ofile_prob_E.readline()
    F = np.array([float(x) for x in line.split()])

    ofile_prob_E.close()
    return bin_centers, F

rc, F = ReadReferenceF("FES-RC.dat")
plt.figure()
plt.plot(rc,F,marker="o")
plt.savefig("FES-RC.png", bbox_inches="tight")
rc, H = ReadReferenceF("HIST-RC.dat")
plt.figure()
plt.plot(rc,H,marker="o")
plt.savefig("HIST-RC.png", bbox_inches="tight")
