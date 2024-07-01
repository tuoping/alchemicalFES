import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os,sys
import time
from copy import deepcopy

alpha_min_kBT = 10
seq_dim = (6,6)
num_batches=1
epoch1=int(sys.argv[1])
T1=1.0

val_dirname  = {
    T1: "../"
}
ref_dirname = "/nfs/scistore14/chenggrp/ptuo/NeuralRG/dirichlet-flow-matching-test3/data/ising-latt%dx%d-T4.0/latt%dx%d/"%(*seq_dim, *seq_dim)

global_s_time = time.time()


import torch
def RC(logits):
    assert logits.shape[-1] == 2
    B = logits.shape[0]
    RC = torch.sum(logits*torch.tensor([-1,1])[None,None,None,:], dim=-1)
    RC = torch.sum(RC.reshape(B, -1), dim=-1)
    return RC.reshape(-1,1)

def reversekde(rc_trajs, x_grid, bandwidth=2.):
        """
        Kernel Density Estimation (KDE) using a Gaussian kernel.

        Parameters:
        - data: Tensor containing the data points (1-dimensional).
        - x_grid: Tensor containing the grid points for which to compute the KDE.
        - bandwidth: Bandwidth parameter for the Gaussian kernel (default=0.5).

        Returns:
        - Tensor of shape (len(x_grid),) containing the estimated density values.
        """
        # Initialize density estimates
        density = torch.zeros_like(x_grid)
        # np.savetxt("rc-logits_train.dat", rc_trajs, delimiter=" ")
        # Compute KDE
        num_Inf = 0
        for i, x in enumerate(x_grid):
            # Kernel function (Gaussian kernel)
            kernel = torch.exp(-0.5 * ((rc_trajs.to(torch.float64) - x) / bandwidth)**2) / torch.sqrt(2 * torch.tensor(3.141592653589793))
            # Sum over all data points
            density[i] = torch.sum(kernel) / (bandwidth * torch.sqrt(2 * torch.tensor(3.141592653589793)))
            if torch.isinf(density[i]):
                num_Inf+=1
        np.savetxt("density-logits_train.dat", density, delimiter=" ")
        norm_density = density/torch.sum(density)
        print("HIST:: ", num_Inf)
        # print(norm_density)
        idxF = np.where(norm_density>0)
        F = -torch.log(norm_density[idxF])
        return F,norm_density.to(torch.float16),idxF

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
def loadmodelprediction(_dirname, epoch, num_batches, fileheader="logits"):
    dirname = _dirname+"/epoch%02d_sample%d"%(epoch,num_batches)
    f_logits_t = sorted(glob.glob(os.path.join(dirname, fileheader+"_val_inttime*")))
    print(">>> Reading model predictions from: ", dirname)

    logits_t = [np.load(f).astype(np.float16) for f in f_logits_t]
    t = [float(x.replace(os.path.join(dirname, fileheader+"_val_inttime"), "").replace(".npy",""))-1 for x in f_logits_t]
    print("Integration time=",t)

    '''
    for ii in range(num_batches):
        if ii == 0:
            print("        ", len(logits_t), [logits_t[i].shape for i in range(len(logits_t))])
            continue
        dirname = _dirname+"/epoch%d_sample%d"%(epoch,ii+1)
        _f_logits_t = sorted(glob.glob(os.path.join(dirname, fileheader+"_val_inttime*")))
        _logits_t = [np.load(f).astype(np.float16) for f in _f_logits_t]
        print("        ", ii+1,len(_logits_t), [_logits_t[i].shape for i in range(len(_f_logits_t))])
        logits_t = [np.concatenate([logits_t[i], _logits_t[i]], axis=0) for i in range(len(_f_logits_t))]
    '''
    return logits_t,t

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


def run_RC_train(T, epoch):
    ### load model predictions
    s_time = time.time()
    # logits_t = loadmodelprediction(val_dirname[T], epoch, num_batches)
    logits = np.load("logits_train_step0.npy")
    e_time = time.time()
    print("Time for loading predictions of model(kBT=%.01f):: "%T, e_time-s_time)
    s_time = e_time
    # for ii in range(len(logits_t)):
    rc = RC(torch.from_numpy(logits))
    bin_centers = np.array([(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)])
    F,H = kde(rc,torch.from_numpy(bin_centers),1.)
    plt.figure()
    plt.hist(rc.numpy(), density=True)
    plt.plot(bin_centers, H.numpy())
    plt.savefig("HIST-RC-TRAIN.png", bbox_inches="tight")
    plt.figure()

    plt.plot(bin_centers, F.numpy())
    plt.savefig("F-RC-TRAIN.png", bbox_inches="tight")


def run_RC_val(T, epoch):
    ### load model predictions
    s_time = time.time()
    logits_t = loadmodelprediction(val_dirname[T], epoch, num_batches)
    e_time = time.time()
    print("Time for loading predictions of model(kBT=%.01f):: "%T, e_time-s_time)
    s_time = e_time
    for ii in range(len(logits_t)):
        rc = RC(torch.from_numpy(logits_t[ii]))
        bin_centers = np.array([(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)])
        F,H = kde(rc,torch.from_numpy(bin_centers),1.)
        plt.figure()
        plt.hist(rc.numpy(), density=True)
        plt.plot(bin_centers, H.numpy())
        plt.savefig("HIST-RC-t%d.png"%ii, bbox_inches="tight")
        plt.figure()

        plt.plot(bin_centers, F.numpy())
        plt.savefig("F-RC-t%d.png"%ii, bbox_inches="tight")

def run_statistics_rcrew(T, epoch):
    ### load model predictions
    s_time = time.time()
    logits_t = loadmodelprediction(val_dirname[T], epoch, num_batches)
    seq_t = logits2seq(logits_t)
    e_time = time.time()
    print("Time for loading predictions of model(kBT=%.01f):: "%T, e_time-s_time)
    s_time = e_time

    ### loading model predictions at T and calculating the potential energy and its statistics.
    for ii in range(len(seq_t)):
        ofile_Prob = open("REW-PROB-MAGN-kBT%.2f-t%d.dat"%(T,ii), "wb")
        ofile_F = open("REW-F-MAGN-kBT%.2f-t%d.dat"%(T,ii), "wb")
        
        print(">>> PROCESSING:: ")
        hist_E, bin_centers_E, P_E, F_E, idxF_E = histvar(seq_t[ii], Ising_magnetization, bins)
        ### RC and KDE
        rc = RC(torch.from_numpy(logits_t[ii]))
        Fbias,Hbias,idxF_bias = reversekde(rc,torch.from_numpy(bin_centers_E),1.)
        ### Reweight
        assert Hbias.shape == P_E.shape
        # rew_P_E = P_E*Hbias.numpy()
        # idxF_E2 = np.where(rew_P_E > 0)
        # print("Sum of Histogram =", P_E.sum())
        # print("Sum of reweighted Histogram = ", rew_P_E[idxF_E2].sum())
        # print(rew_P_E)
        # rew_F = -np.log(rew_P_E[idxF_E2]/rew_P_E[idxF_E2].sum())
        idxF_E2 = np.intersect1d(idxF_E, idxF_bias)
        rew_F = -np.log(P_E[idxF_E2])+np.log(Hbias.numpy()[idxF_E2])
        

        # np.savetxt(ofile_Prob, bin_centers_E.reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="BIN CENTERS kBT=%.2f"%T)
        # np.savetxt(ofile_Prob, rew_P_E.reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="PROB kBT=%.2f"%T)
        np.savetxt(ofile_F, bin_centers_E[idxF_E].reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="BIN CENTERS kBT=%.2f"%T)
        np.savetxt(ofile_F, rew_F.reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="F kBT=%.2f"%T)

        plt.figure()
        plt.scatter(bin_centers_E[idxF_E], F_E, label="Model prediction", marker="o", c="green")
        if T in Reference_dict:
            plt.plot(Reference_dict[T][0], Reference_dict[T][1], c="green", label="Ground truth")
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel("Magnetization", fontdict={"size":14})
        plt.ylabel("Free energy ($k_BT$)", fontdict={"size":14})
        plt.savefig("F-MAGN-kBT%.2f-t%d.png"%(T,ii), bbox_inches="tight")

        plt.figure()
        plt.scatter(bin_centers_E[idxF_bias], Fbias, label="RC bias", marker="o", c="green")
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel("Magnetization", fontdict={"size":14})
        plt.ylabel("Free energy ($k_BT$)", fontdict={"size":14})
        plt.savefig("BIAS-MAGN-kBT%.2f-t%d.png"%(T,ii), bbox_inches="tight")

        plt.figure()
        plt.scatter(bin_centers_E[idxF_E2], rew_F, label="Model prediction", marker="o", c="green")
        if T in Reference_dict:
            plt.plot(Reference_dict[T][0], Reference_dict[T][1], c="green", label="Ground truth")
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel("Magnetization", fontdict={"size":14})
        plt.ylabel("Free energy ($k_BT$)", fontdict={"size":14})
        plt.savefig("REW-F-MAGN-kBT%.2f-t%d.png"%(T,ii), bbox_inches="tight")

        ofile_Prob.flush()
        ofile_F.flush()
        ofile_Prob.close()
        ofile_F.close()
    e_time = time.time()
    print("Time for processing:: ", e_time-s_time)


def run_statistics_xt(T, epoch):
    ### load model predictions
    s_time = time.time()
    logits_t, t_integration = loadmodelprediction(val_dirname[T], epoch, num_batches, fileheader="xt")
    seq_t = logits2seq(logits_t)
    e_time = time.time()
    print("Time for loading predictions of model(kBT=%.01f):: "%T, e_time-s_time)
    s_time = e_time

    ### loading model predictions at T and calculating the potential energy and its statistics.
    line_color = [plt.colormaps["gnuplot"](float(20-1-i)/float(20)) for i in range(20)]
    plt.figure()
    for ii in range(0, len(seq_t), 2):
        ofile_Prob = open("xt-PROB-MAGN-kBT%.2f-t%d.dat"%(T,ii), "wb")
        ofile_F = open("xt-F-MAGN-kBT%.2f-t%d.dat"%(T,ii), "wb")
        # plt.figure()
        print(">>> PROCESSING:: ")
        hist_E, bin_centers_E, P_E, F_E, idxF_E = histvar(seq_t[ii], Ising_magnetization, bins)
        np.savetxt(ofile_Prob, bin_centers_E.reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="BIN CENTERS kBT=%.2f"%T)
        np.savetxt(ofile_Prob, P_E.reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="PROB kBT=%.2f"%T)
        np.savetxt(ofile_F, bin_centers_E[idxF_E].reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="BIN CENTERS kBT=%.2f"%T)
        np.savetxt(ofile_F, F_E.reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="F kBT=%.2f"%T)
        plt.scatter(bin_centers_E[idxF_E], F_E, label="$k_BT=%f$"%(alpha_min_kBT/(t_integration[ii]+1e-5)), c=line_color[ii])
        # if T in Reference_dict:
        #     plt.plot(Reference_dict[T][0], Reference_dict[T][1], c="green", label="Ground truth")
        
        # plt.tick_params(axis='both', which='major', labelsize=14)
        # plt.xlabel("Magnetization", fontdict={"size":14})
        # plt.ylabel("Free energy ($k_BT$)", fontdict={"size":14})
        # plt.savefig("F-MAGN-kBT%.2f-t%d.png"%(T,ii), bbox_inches="tight")
        ofile_Prob.flush()
        ofile_F.flush()
        ofile_Prob.close()
        ofile_F.close()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel("Magnetization", fontdict={"size":14})
    plt.ylabel("Free energy ($k_BT$)", fontdict={"size":14})
    plt.savefig("xt-F-MAGN.png", bbox_inches="tight")
    e_time = time.time()
    print("Time for processing:: ", e_time-s_time)

def run_statistics(T, epoch):
    ### load model predictions
    s_time = time.time()
    logits_t, t_integration = loadmodelprediction(val_dirname[T], epoch, num_batches)
    seq_t = logits2seq(logits_t)
    e_time = time.time()
    print("Time for loading predictions of model(kBT=%.01f):: "%T, e_time-s_time)
    s_time = e_time

    ### loading model predictions at T and calculating the potential energy and its statistics.
    line_color = [plt.colormaps["gnuplot"](float(20-1-i)/float(20)) for i in range(20)]
    plt.figure()
    for ii in range(0, len(seq_t), 2):
        ofile_Prob = open("PROB-MAGN-kBT%.2f-t%d.dat"%(T,ii), "wb")
        ofile_F = open("F-MAGN-kBT%.2f-t%d.dat"%(T,ii), "wb")
        # plt.figure()
        print(">>> PROCESSING:: ")
        hist_E, bin_centers_E, P_E, F_E, idxF_E = histvar(seq_t[ii], Ising_magnetization, bins)
        np.savetxt(ofile_Prob, bin_centers_E.reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="BIN CENTERS kBT=%.2f"%T)
        np.savetxt(ofile_Prob, P_E.reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="PROB kBT=%.2f"%T)
        np.savetxt(ofile_F, bin_centers_E[idxF_E].reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="BIN CENTERS kBT=%.2f"%T)
        np.savetxt(ofile_F, F_E.reshape([1,-1]), fmt="%4.4e", delimiter=" ", header="F kBT=%.2f"%T)
        plt.scatter(bin_centers_E[idxF_E], F_E, label="$k_BT=%f$"%(alpha_min_kBT/(t_integration[ii]+1e-5)), c=line_color[ii])
        # if T in Reference_dict:
        #     plt.plot(Reference_dict[T][0], Reference_dict[T][1], c="green", label="Ground truth")
        
        # plt.tick_params(axis='both', which='major', labelsize=14)
        # plt.xlabel("Magnetization", fontdict={"size":14})
        # plt.ylabel("Free energy ($k_BT$)", fontdict={"size":14})
        # plt.savefig("F-MAGN-kBT%.2f-t%d.png"%(T,ii), bbox_inches="tight")
        ofile_Prob.flush()
        ofile_F.flush()
        ofile_Prob.close()
        ofile_F.close()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel("Magnetization", fontdict={"size":14})
    plt.ylabel("Free energy ($k_BT$)", fontdict={"size":14})
    plt.savefig("F-MAGN.png", bbox_inches="tight")
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


run_statistics_xt(T1, epoch1)
run_statistics(T1, epoch1)
# run_RC_val(T1, epoch1)

print("Total wall time:: ", time.time()-global_s_time)
