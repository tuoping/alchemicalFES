import numpy as np 
import matplotlib.pyplot as plt
seq_dim=(24,24)

magn = np.arange(-np.prod(seq_dim), np.prod(seq_dim)+1, 2)
bins=np.linspace(magn[0]-1, magn[-1]+1, np.prod(seq_dim)+1+1)

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


def calculateError(free_energys_list_, num_samples=4):
    free_energys_list_ = np.array(free_energys_list_)
    std_free = np.std(free_energys_list_, axis=0)
    standard_error = std_free / np.sqrt(num_samples)
    t_critical = 1.96
    margin_of_error = t_critical * standard_error
    free_energies = np.mean(free_energys_list_, axis=0)
    return free_energies, margin_of_error

from functools import reduce
def histvar(seq, varfunc, bins, num_samples):
    var = varfunc(seq)
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
    P_res = calculateError(P_all)
    F_res = calculateError(F_all)
    return hist, bin_centers_all[0], P_res, F_res, np.array(idxF_res)

logits_t, diffusion_t = loadmodelprediction("./", num_batches=1)
seq_t = logits2seq(logits_t)

ofile_fes = open("FES-MAGN.dat", "wb")
for idx_t, i in enumerate(range(len(diffusion_t))):
    # if diffusion_t[i] >= 1.0:
    #     break
    hist_E, bin_centers_E, P_E, F_E, idxF_E = histvar(seq_t[i], Ising_magnetization, bins, num_samples=4)
    np.savetxt(ofile_fes, bin_centers_E[idxF_E].reshape(1,-1), delimiter=" ", header=f"alpha-1={diffusion_t[i]}; Magnetization")
    np.savetxt(ofile_fes, F_E[0].reshape(1,-1), delimiter=" ", header=f"alpha-1={diffusion_t[i]}; FES")
    np.savetxt(ofile_fes, F_E[1].reshape(1,-1), delimiter=" ", header=f"alpha-1={diffusion_t[i]}; ERROR of FES")
ofile_fes.close()
