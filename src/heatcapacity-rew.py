import numpy as np 
import matplotlib.pyplot as plt
seq_dim=(24,24)

min_E = -np.prod(seq_dim)*2.
max_E = 0
num_bins = int((max_E-min_E)/2)
bins = np.linspace(min_E-1./num_bins, max_E+1./num_bins, num_bins+2)

import sys
kBT = float(sys.argv[1])

import os
import glob
def loadmodelprediction(_dirname, epoch=None, num_batches=1, file_header="logits"):
    dirname = _dirname
    f_logits_t = glob.glob(os.path.join(dirname, file_header+"_val_inttime*"))
    print(">>> Reading model predictions from: ", dirname)

    logits_t = np.array([np.load(f).astype(np.float16) for f in f_logits_t])
    for i in range(1, num_batches):
        dirname_2 = f'../g0_{i}'
        f_logits_t_2 = glob.glob(os.path.join(dirname_2, file_header+"_val_inttime*"))
        logits_t_2 = np.array([np.load(f).astype(np.float16) for f in f_logits_t_2])
        print(logits_t.shape, logits_t_2.shape)
        logits_t = np.concatenate([logits_t, logits_t_2], axis=1)
    diffusion_t = np.array([float(x.replace(os.path.join(dirname, file_header+"_val_inttime"), "").replace(".npy",""))-1 for x in f_logits_t])

    idx_order = np.argsort(diffusion_t)
    logits_t = logits_t[idx_order]
    diffusion_t = diffusion_t[idx_order]
    return logits_t, diffusion_t

def logits2seq(logits_t):
    seq_t = []
    for logits in logits_t:
        seq = np.argmax(logits, axis=-1)
        seq[np.where(seq==0)] = -1
        seq_t.append(seq.reshape(-1,*seq_dim))
    return seq_t

def Ising_magnetization(seq):
    data = np.sum(seq.reshape([-1,np.prod(seq_dim)]), axis=-1)
    return data

def pbc(i,L=seq_dim[0]):
    assert i>=-1 and i<=L
    if i-L == 0:
        return 0
    elif i == -1:
        return L-1
    else:
        return i
    
from copy import deepcopy
def ising_boltzman_prob_nn(seq, J=1, kBT=1.0):
    shape = seq.shape
    # spins = seq.clone().detach()
    spins = deepcopy(seq)
    spins[np.where(spins==0)]=-1
    B,H,W = shape
    E = np.zeros(B)
    for i in range(H):
        for j in range(W):
            E += -spins[:,i,j]*spins[:,pbc(i-1),j]*J
            E += -spins[:,i,j]*spins[:,pbc(i+1),j]*J
            E += -spins[:,i,j]*spins[:,i,pbc(j-1)]*J
            E += -spins[:,i,j]*spins[:,i,pbc(j+1)]*J

    E /= 2
    # prob = np.exp(-E/kBT)
    return E/kBT

def calculateError(free_energys_list_, num_samples=4):
    free_energys_list_ = np.array(free_energys_list_)
    std_free = np.std(free_energys_list_, axis=0)
    standard_error = std_free / np.sqrt(num_samples)
    t_critical = 1.96
    margin_of_error = t_critical * standard_error
    free_energies = np.mean(free_energys_list_, axis=0)
    return free_energies, margin_of_error

# def expectation_var(seq, varfunc):
#     var = varfunc(seq)
#     expectation_var = (var**2).mean() - (var.mean())**2
#     t_critical = 1.96
#     margin_of_error = t_critical * expectation_var * np.sqrt(2/(len(var)-1))
#     return expectation_var, margin_of_error

def expectation_var(seq, varfunc):
    var = varfunc(seq)
    log_Z = np.logaddexp.reduce(-var/kBT)
    w = np.exp(-var/kBT - log_Z)
    assert np.abs(w.sum()-1.0) < 0.001
    expectation_var = (w*var**2).sum() - ((var*w).sum())**2
    t_critical = 1.96
    margin_of_error = t_critical * expectation_var * np.sqrt(2/(len(var)-1))
    return expectation_var, margin_of_error

from functools import reduce
def histvar(seq, varfunc, bins, num_samples):
    var = varfunc(seq)
    expectation_var = var.mean()
    t_critical = 1.96
    margin_of_error = t_critical * var.std()/np.sqrt(len(var))
    if var.shape[0]<1024*4:
        print("WARNING:: sample not enough to generate reliable error bars")
    P_all = []
    idxF_all = []
    hist_all = []
    bin_centers_all = []
    for ii in range(num_samples):
        hist, bin_edges = np.histogram(var[ii::num_samples], bins=bins)
        bin_centers = np.array([(bin_edges[i]+bin_edges[i+1])/2. for i in range(len(bin_edges)-1)])
        bin_centers_all.append(bin_centers)
        hist_all.append(hist)
        P = hist/np.sum(hist)
        P_all.append(P)
        idxF = np.where(hist>0)
        idxF_all.append(idxF)
        # F = -np.log(P[idxF])
        # F_all.append(F)

    P_all = np.array(P_all)
    hist_all = np.array(hist_all)
    idxF_res = reduce(np.intersect1d, tuple(idxF_all))
    if num_samples == 1:
        idxF_res = idxF_res[0]
    F_all = []
    for ii in range(num_samples):
        F = -np.log(P_all[ii][idxF_res])
        F_all.append(F)
    F_all = np.array(F_all)

    for ii in range(num_samples-1):
        if not np.array_equal(bin_centers_all[ii], bin_centers_all[ii+1]):
            raise Exception("ERROR:: bin_centers not consistant")
    hist = np.mean(hist_all)
    P_res = calculateError(P_all, num_samples=num_samples)
    F_res = calculateError(F_all, num_samples=num_samples)
    return hist, bin_centers_all[0], P_res, F_res, np.array(idxF_res)

logits_t, diffusion_t = loadmodelprediction("./", num_batches=1)
seq_t = logits2seq(logits_t)

'''
magn = Ising_magnetization(seq_t[-1])
pote = ising_boltzman_prob_nn(seq_t[-1])

plt.figure(figsize=(12,5))
plt.subplot(121)
H, xedges, yedges, img = plt.hist2d(magn, pote, bins=100, cmap="Blues", density=True, vmax=0.0004)
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
plt.colorbar()
# plt.scatter(magn_c4[ridx], pote_c4[ridx], marker="*", c="red", s=60)
# plt.ylim((-300, -25))
plt.subplot(122)
# _ = plt.imshow(seq_c4[ridx], cmap='Greys', vmin=0, vmax=1)
# plt.xticks([])
# plt.yticks([])
plt.contour(X, Y, H.T, levels=10, cmap="plasma")
plt.colorbar()
# plt.ylim((-300, -25))
plt.savefig("hist.png")

from scipy.ndimage import gaussian_filter1d

ofile_fes = open("FES-E.dat", "wb")
# ofile_fes_smooth = open("FES-E-smoothed.dat", "wb")
ofile_prob = open("PROB-E.dat", "wb")
for idx_t, i in enumerate(range(len(diffusion_t))):

    hist_E, bin_centers_E, P_E, F_E, idxF_E = histvar(seq_t[i], ising_boltzman_prob_nn, bins, num_samples=4)
    np.savetxt(ofile_fes, bin_centers_E[idxF_E].reshape(1,-1), delimiter=" ", header=f"alpha-1={diffusion_t[i]}; Potential energy (J)")
    np.savetxt(ofile_fes, F_E[0].reshape(1,-1), delimiter=" ", header=f"alpha-1={diffusion_t[i]}; FES")
    np.savetxt(ofile_fes, F_E[1].reshape(1,-1), delimiter=" ", header=f"alpha-1={diffusion_t[i]}; ERROR of FES")

    # sigma_gkernel = 3
    # y_smoothed = gaussian_filter1d(F_E[0], sigma_gkernel)
    # errors_smoothed = gaussian_filter1d(F_E[1], sigma_gkernel)
    # np.savetxt(ofile_fes_smooth, bin_centers_E[idxF_E].reshape(1,-1), delimiter=" ", header=f"alpha-1={diffusion_t[i]}; Potential energy (J)")
    # np.savetxt(ofile_fes_smooth, y_smoothed.reshape(1,-1), delimiter=" ", header=f"alpha-1={diffusion_t[i]}; FES")
    # np.savetxt(ofile_fes_smooth, errors_smoothed.reshape(1,-1), delimiter=" ", header=f"alpha-1={diffusion_t[i]}; ERROR of FES")

    np.savetxt(ofile_prob, bin_centers_E.reshape(1,-1), delimiter=" ", header=f"alpha-1={diffusion_t[i]}; Potential energy (J)")
    np.savetxt(ofile_prob, P_E[0].reshape(1,-1), delimiter=" ", header=f"alpha-1={diffusion_t[i]}; PROB")
    np.savetxt(ofile_prob, P_E[1].reshape(1,-1), delimiter=" ", header=f"alpha-1={diffusion_t[i]}; ERROR of PROB")


ofile_fes.close()
ofile_prob.close()
'''
ofile_expectation = open("Expectation-HeatCapacity-rew.dat", "w")
ofile_expectation.write("DiffusionTime    Cv err\n")
for idx_t, i in enumerate(range(len(diffusion_t))):
    expectation_pote, error = expectation_var(seq_t[i], ising_boltzman_prob_nn)
    ofile_expectation.write("%f    %f  %f\n"%(diffusion_t[i], expectation_pote, error))
