import numpy as np
import matplotlib.pyplot as plt

bs = 32768
def ReadReferenceF(filename):
    ofile_prob_E = open(filename,"r")
    lines = ofile_prob_E.readlines()
    bin_centers = np.array([float(x) for x in lines[1].split()])
    F = []
    for line in lines[3:]:
        F.append([float(x) for x in line.split()])
    ofile_prob_E.close()
    return bin_centers, np.array(F)

rc, F = ReadReferenceF("FES-RC-GT.dat")
plt.figure()
plt.plot(rc,-np.logaddexp.reduce(-F, axis=0),marker="o")
plt.savefig("FES-RC-GT.png", bbox_inches="tight")
rc, H = ReadReferenceF("HIST-RC-GT.dat")
plt.figure()
plt.plot(rc,np.sum(H, axis=0),marker="o")
plt.savefig("HIST-RC-GT.png", bbox_inches="tight")

rc, F = ReadReferenceF("FES-RC.dat")
plt.figure()
plt.plot(rc,-np.logaddexp.reduce(-F, axis=0),marker="o")
plt.savefig("FES-RC.png", bbox_inches="tight")
rc, H = ReadReferenceF("HIST-RC.dat")
plt.figure()
plt.plot(rc,np.sum(H, axis=0),marker="o")
plt.savefig("HIST-RC.png", bbox_inches="tight")

