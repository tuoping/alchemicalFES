import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os
import time
from copy import deepcopy

seq_dim = (6,6)
num_batches=36
epoch1=152
epoch2=101
T1=2.0
T2=2.2

val_dirname  = {
    T1: "/nfs/scistore14/chenggrp/ptuo/NeuralRG/dirichlet-flow-matching-test2/logs-dir-ising/latt6x6T%.01f/kernel3x3_timeembed/finetune9/val_baseline_latt%dx%d"%(T1,*seq_dim),
    T2: "/nfs/scistore14/chenggrp/ptuo/NeuralRG/dirichlet-flow-matching-test2/logs-dir-ising/latt6x6T%.01f/kernel3x3_timeembed/val_baseline_latt%dx%d"%(T2,*seq_dim),
}
ref_dirname = "/nfs/scistore14/chenggrp/ptuo/NeuralRG/dirichlet-flow-matching-test3/data/ising-latt%dx%d-T4.0/latt%dx%d/"%(*seq_dim, *seq_dim)

global_s_time = time.time()
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

def Ising_magnetization(seq):
    data = np.sum(seq.reshape([-1,np.prod(seq_dim)]), axis=-1)
    return data


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
    print(">>> Reading model predictions from: ", dirname)

    logits_t = [np.load(f).astype(np.float16) for f in [f_logits_t[0], f_logits_t[-1]]]

    for ii in range(num_batches):
        if ii == 0:
            print("        ", len(logits_t), [logits_t[i].shape for i in range(len(logits_t))])
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

def getIntersection(list1, list2):
    # List to store the indices
    indices = []
    # Loop through the first list
    for index1, element in enumerate(list1):
        if element in list2:
            index2 = list2.index(element)
            indices.append([index1, index2])
    indices = np.array(indices).T
    intersection_list = np.array(list1)[indices[0]]
    assert np.sum(np.equal(intersection_list, np.array(list2)[indices[1]])) == len(intersection_list)
    return intersection_list, indices[0], indices[1]

magn = np.arange(-np.prod(seq_dim), np.prod(seq_dim)+1, 2)
bins=np.linspace(magn[0]-1, magn[-1]+1, np.prod(seq_dim)+1+1)

Reference_dict = ReadReferenceF(os.path.join(ref_dirname, "F-MAGN-REF.dat"))

def run_statistics(T, epoch):
    ### load model predictions
    s_time = time.time()
    logits_t = loadmodelprediction(val_dirname[T], epoch, num_batches)
    seq_t = logits2seq(logits_t)
    e_time = time.time()
    print("Time for loading predictions of model(kBT=%.01f):: "%T, e_time-s_time)
    s_time = e_time

    ### loading model predictions at T and calculating the potential energy and its statistics.
    ofile_Prob = open("PROB-MAGN-kBT%.2f.dat"%T, "wb")
    ofile_F = open("F-MAGN-kBT%.2f.dat"%T, "wb")
    plt.figure()
    print(">>> PROCESSING:: ")
    hist_E, bin_centers_E, P_E, F_E, idxF_E = histvar(seq_t[-1], Ising_magnetization, bins)
    np.savetxt(ofile_Prob, bin_centers_E.reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="BIN CENTERS kBT=%.2f"%T)
    np.savetxt(ofile_Prob, P_E.reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="PROB kBT=%.2f"%T)
    np.savetxt(ofile_F, bin_centers_E[idxF_E].reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="BIN CENTERS kBT=%.2f"%T)
    np.savetxt(ofile_F, F_E.reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="F kBT=%.2f"%T)
    plt.scatter(bin_centers_E[idxF_E], F_E, label="Model prediction", marker="o", c="green")
    if T in Reference_dict:
        plt.plot(Reference_dict[T][0], Reference_dict[T][1], c="green", label="Ground truth")
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel("Magnetization", fontdict={"size":14})
    plt.ylabel("Free energy ($k_BT$)", fontdict={"size":14})
    plt.savefig("F-MAGN-kBT%.2f.png"%T, bbox_inches="tight")
    ofile_Prob.flush()
    ofile_F.flush()
    ofile_Prob.close()
    ofile_F.close()
    e_time = time.time()
    print("Time for processing:: ", e_time-s_time)

def run_interpolate_DOS():
    ### calculating DOS from the FES of two temperatures
    s_time = time.time()
    Prediction_dict_T1 = ReadReferenceF("F-MAGN-kBT%.2f.dat"%T1)
    Prediction_dict_T2 = ReadReferenceF("F-MAGN-kBT%.2f.dat"%T2)
    print(">>> Inpterpolating from :: ")
    print(Prediction_dict_T1)
    print(Prediction_dict_T2)
    bin_centers, idxT1, idxT2 = getIntersection(list(Prediction_dict_T1[T1][0]), list(Prediction_dict_T2[T2][0]))
    beta1 = 1/T1
    beta2 = 1/T2
    DOS = Prediction_dict_T1[T1][1][idxT1]+beta1/(beta1-beta2)*(-Prediction_dict_T1[T1][1][idxT1]+Prediction_dict_T2[T2][1][idxT2])

    ofile_DOS = open("DOS-MAGN-interpolateT%.2fT%.2f.dat"%(T1,T2),"wb")
    np.savetxt(ofile_DOS, bin_centers.reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="BIN CENTERS interpolated from kBT= %.2f %.2f"%(T1,T2))
    np.savetxt(ofile_DOS, DOS.reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="DOS interpolated from kBT= %.2f %.2f"%(T1,T2))
    ofile_DOS.flush()
    ofile_DOS.close()

    plt.figure()
    plt.plot(bin_centers, DOS, label="DOS", c="red")
    plt.scatter(bin_centers, Prediction_dict_T1[T1][1][idxT1], c="k", label="$k_BT=%2f$"%T1)
    if T1 in Reference_dict:
        plt.plot(Reference_dict[T1][0], Reference_dict[T1][1], c="k")
    plt.scatter(bin_centers, Prediction_dict_T2[T2][1][idxT2], c="green", label="$k_BT=%2f$"%T2)
    if T2 in Reference_dict:
        plt.plot(Reference_dict[T2][0], Reference_dict[T2][1], c="green")

    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel("Magnetization", fontdict={"size":14})
    plt.ylabel("Free energy ($k_BT$)", fontdict={"size":14})
    plt.savefig("DOS-MAGN-interpolateT%.2fT%.2f.png"%(T1, T2), bbox_inches="tight")

def run_interpolate_FES(T3):
    ### Interpolating FES of any temperature from DOS and T1
    s_time = time.time()
    DOS_dict = ReadReferenceF("DOS-MAGN-interpolateT%.2fT%.2f.dat"%(T1,T2))
    DOS = list(DOS_dict.values())[0]
    Prediction_dict_T3 = ReadReferenceF("F-MAGN-kBT%.2f.dat"%T3)
    FES_T3 = list(Prediction_dict_T3.values())[0]
    bin_centers, idxDOS, idxT3 = getIntersection(list(DOS[0]), list(FES_T3[0]))
    assert len(idxDOS) == len(DOS[0])

    plt.figure()
    line_color = [plt.colormaps["gnuplot"](float(i)/float(10)) for i in range(10)]
    plt.scatter(DOS[0], DOS[1], c="r")
    ofile_F = open("F-MAGN-interpolateDOST%.2fT%.2fFT%.2f.dat"%(T1,T2,T3),"wb")
    np.savetxt(ofile_F, bin_centers.reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="BIN CENTERS interpolated from kBT= %.2f %.2f"%(T1,T2))
    Expectation_E_list = []
    for idx_jj,jj in enumerate(list(Reference_dict.keys())):
        _F = (FES_T3[1][idxT3]-DOS[1])*T3/jj + DOS[1]
        P = np.exp(-_F)
        print(jj, np.sum(P))
        P = P/np.sum(P)
        F = _F+np.log(np.sum(P))
        np.savetxt(ofile_F, F.reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="FES interpolated from DOS(kBT=%.2f %.2f) and F(kBT=%.2f)"%(T1,T2,T3))
        # P = np.exp(-F)
        # print(jj, np.sum(P))
        # P = P/np.sum(P)
        Expectation_E_list.append([jj, np.sum(bin_centers*P)])
        if jj == T3:
            plt.plot(Reference_dict[jj][0], Reference_dict[jj][1], c="green")
            plt.scatter(bin_centers, F, c="green", label="$k_BT=%.2f$"%jj) 
        else:
            plt.plot(Reference_dict[jj][0], Reference_dict[jj][1], c=line_color[idx_jj])
            plt.scatter(bin_centers, F, c=line_color[idx_jj], label="$k_BT=%.2f$"%jj)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel("Magnetization", fontdict={"size":14})
    plt.ylabel("Free energy ($k_BT$)", fontdict={"size":14})
    plt.savefig("F-MAGN-interpolateDOST%.2fT%.2fFT%.2f.png"%(T1,T2,T3), bbox_inches="tight")
    ofile_F.flush()
    ofile_F.close()
    Expectation_E_list = np.array(Expectation_E_list)
    np.savetxt("Expectation-MAGN-interpolateDOST%.2fT%.2fFT%.2f.dat"%(T1,T2,T3), Expectation_E_list, fmt="%4.4e", delimiter=" ", header="kBT Expectation_E")


def plot_statistics(T):
    ### load model predictions
    Prediction_dict_T = ReadReferenceF("F-MAGN-kBT%.2f.dat"%T)
    plt.figure()
    plt.scatter(Prediction_dict_T[T][0], Prediction_dict_T[T][1], label="Model prediction", marker="o", c="green")
    if T in Reference_dict:
        plt.plot(Reference_dict[T][0], Reference_dict[T][1], c="green", label="Ground truth")
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel("Magnetization", fontdict={"size":14})
    plt.ylabel("Free energy ($k_BT$)", fontdict={"size":14})
    plt.savefig("F-MAGN-kBT%.2f.png"%T, bbox_inches="tight")

'''
def plot_kBT_expectation(T3):
    ### plot kBT-Expectation_E
    s_time = time.time()
    Expectation_E = np.loadtxt("Expectation-MAGN-interpolateDOST%.2fT%.2fFT%.2f.dat"%(T1,T2,T3), skiprows=1)
    Expectation_E_REF = np.loadtxt(os.path.join(ref_dirname, "Expectation-MAGN-REF.dat"), skiprows=1)
    plt.figure()
    idx_order_ref = np.argsort(Expectation_E_REF[:,0])
    plt.plot(Expectation_E_REF[:,0][idx_order_ref], Expectation_E_REF[:,1][idx_order_ref], c="k")
    plt.scatter(Expectation_E[:,0], Expectation_E[:,1], c="blue")
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel("$k_BT$", fontdict={"size":14})
    plt.ylabel("Energy", fontdict={"size":14})
    plt.savefig("E-T-interpolateDOST%.2fT%.2fFT%.2f.png"%(T1,T2,T3), bbox_inches="tight")
'''

plot_statistics(T2)
# run_statistics(T1, epoch1)

print("Total wall time:: ", time.time()-global_s_time)
