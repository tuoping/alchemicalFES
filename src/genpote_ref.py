import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os
import time
from copy import deepcopy


seq_dim = (6,6)

s_time = time.time()
global_s_time = s_time

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
    prob = np.exp(-E/kBT)
    return prob, E/kBT


def spin_structure_factor(seq):
    shape = seq.shape
    # spins = seq.clone().detach()
    spins = deepcopy(seq)
    spins[np.where(spins==0)]=-1
    B,H,W = shape
    E = np.zeros(B)
    for i in range(H):
        for j in range(W):
            # for m in range(H):
            #     for n in range(W):
            #         E += spins[:,i,j]*spins[:,m,n]
            E += spins[:,i,j]*(np.sum(spins.reshape([-1,np.prod(seq_dim)]), axis=-1))

    E /= (np.prod(seq_dim)*np.prod(seq_dim))
    return E

import glob
def loadmodelprediction(_dirname, epoch, num_batches):
    dirname = _dirname+"/epoch%d_sample%d"%(epoch,1)
    f_logits_t = sorted(glob.glob(os.path.join(dirname, "logits_val_step0_inttime*")))
    print(">>> Reading model predictions from: ", _dirname)

    logits_t = [np.load(f).astype(np.float16) for f in [f_logits_t[0], f_logits_t[-1]]]

    for ii in range(num_batches):
        if ii == 0:
            print(len(logits_t), [logits_t[i].shape for i in range(len(logits_t))])
            continue
        dirname = _dirname+"/epoch%d_sample%d"%(epoch,ii+1)
        _f_logits_t = sorted(glob.glob(os.path.join(dirname, "logits_val_step0_inttime*")))
        _logits_t = [np.load(f).astype(np.float16) for f in [_f_logits_t[0], _f_logits_t[-1]]]
        print("        ", ii+1,len(_logits_t), [_logits_t[i].shape for i in range(2)])
        logits_t = [np.concatenate([logits_t[i], _logits_t[i]], axis=0) for i in range(2)]
    return logits_t

def logits2seq(logits_t):
    seq_t = []
    for logits in logits_t:
        seq = np.argmax(logits, axis=-1)
        seq[np.where(seq==0)] = -1
        seq_t.append(seq.reshape(-1,*seq_dim))
    return seq_t

def histvar(seq, varfunc, bins):
    var = varfunc(seq)
    hist, bin_edges = np.histogram(var, bins=bins)
    bin_centers = np.array([(bin_edges[i]+bin_edges[i+1])/2. for i in range(len(bin_edges)-1)])
    P = hist/np.sum(hist)
    idxF = np.where(hist>0)
    F = -np.log(P[idxF])
    return hist, bin_centers, P, F, idxF

def run_loading(Tlist, dumping_mode="a"):
    min_E = -np.prod(seq_dim)*2.
    max_E = np.prod(seq_dim)*2.*3./4.
    num_bins = int((max_E-min_E)/2)
    bins = np.linspace(min_E-1./num_bins, max_E+1./num_bins, num_bins+2)
    ofile_Prob_ref = open("PROB-E-REF.dat", dumping_mode+"b")
    ofile_F_ref = open("F-E-REF.dat", dumping_mode+"b")
    plt.figure()
    line_color = [plt.colormaps["gnuplot"](float(i)/float(len(Tlist))) for i in range(len(Tlist))]

    s_time = time.time()
    ### loading MC data and calculating the potential energy and its statistics
    seq_ref_list = {}
    for idx_jj, jj in enumerate(Tlist):
        print(">>> LOADING REF:: ",jj)
        seq_ref = np.load("./buffer-S%.2f.npy"%(jj)).astype(np.float16).reshape(-1,*seq_dim)
        seq_ref_list[jj] = (seq_ref)
        e_time = time.time()
        print("Time for loading MC data:: ", e_time-s_time)
        s_time = e_time

        print(">>> PROCESSING REF:: ",jj)
        hist_E, bin_centers_E, P_E, F_E, idxF_E = histvar(seq_ref_list[jj], ising_boltzman_prob_nn, bins)
        if idx_jj == 0:
            np.savetxt(ofile_Prob_ref, bin_centers_E.reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="BIN CENTERS: ")
        np.savetxt(ofile_Prob_ref, P_E.reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="PROB kBT=%.2f"%jj)
        np.savetxt(ofile_F_ref, bin_centers_E[idxF_E].reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="BIN CENTERS: ")
        np.savetxt(ofile_F_ref, F_E.reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="F kBT=%.2f"%jj)
        ofile_Prob_ref.flush()
        ofile_F_ref.flush()
        plt.plot(bin_centers_E[idxF_E], F_E, c=line_color[idx_jj], label="$k_BT=%.1f$"%jj, marker="o")
        e_time = time.time()
        print("Time for processing MC data:: ", e_time-s_time)
        
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel("Potential energy", fontdict={"size":14})
    plt.ylabel("Free energy ($k_BT$)", fontdict={"size":14})
    plt.savefig("F-E-REF.png", bbox_inches="tight")


def run_expectation(Tlist, dumping_mode="a"):
    ### calculating expectation of energy from probabilities of energy
    Expectation_E_list = []
    ofile_prob_E = open("PROB-E-REF.dat","r")
    ofile_prob_E.readline()
    line = ofile_prob_E.readline()
    bin_centers = np.array([float(x) for x in line.split()])
    for idx_jj, jj in enumerate(Tlist):
        ofile_prob_E.readline()
        line = ofile_prob_E.readline()
        P = np.array([float(x) for x in line.split()])
        print(jj,np.sum(P),P)
        Expectation_E = np.sum(P*bin_centers)
        Expectation_E_list.append([jj, Expectation_E])
    ofile_prob_E.close()

    ofile_expectation_E = open("Expectation-E-REF.dat",dumping_mode+"b")
    np.savetxt(ofile_expectation_E, Expectation_E_list, header="kBT Expectation_E")
    ofile_expectation_E.close()

    Expectation_E_list = np.array(Expectation_E_list)
    plt.figure()
    plt.plot(Expectation_E_list[:,0], Expectation_E_list[:,1], marker="o")
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel("Energy", fontdict={"size":14})
    plt.ylabel("$k_BT$", fontdict={"size":14})
    plt.savefig("E-T-REF.png", bbox_inches="tight")


run_loading([1.6])
run_expectation([1.6])
