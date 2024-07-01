import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os
import time
from copy import deepcopy

seq_dim = (6,6)
num_batches=1
epoch=101
T1=2.0

global_s_time = time.time()


import glob
def loadmodelprediction(_dirname, epoch, num_batches):
    dirname = _dirname+"/epoch%d_sample%d"%(epoch,1)
    f_logits_t = sorted(glob.glob(os.path.join(dirname, "logits_val_step0_inttime*")))
    print(">>> Reading model predictions from: ", _dirname)

    logits_t = [np.load(f).astype(np.float16) for f in f_logits_t]

    for ii in range(num_batches):
        if ii == 0:
            print("        ", len(logits_t), [logits_t[i].shape for i in range(len(logits_t))])
            continue
        dirname = _dirname+"/epoch%d_sample%d"%(epoch,ii+1)
        _f_logits_t = sorted(glob.glob(os.path.join(dirname, "logits_val_step0_inttime*")))
        _logits_t = [np.load(f).astype(np.float16) for f in _f_logits_t]
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


def plot_seq(T):
    ### load model predictions
    s_time = time.time()
    logits_t = loadmodelprediction("../../../../latt4x4T%.01f/kernel3x3_timeembed/val_baseline_latt%dx%d"%(T, seq_dim[0], seq_dim[1]), epoch, num_batches)
    seq_t = logits2seq(logits_t)
    e_time = time.time()
    print("Time for loading predictions of model(kBT=%.01f):: "%T, e_time-s_time)
    s_time = e_time

    fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(30, 12))

    for ii in np.arange(0,len(seq_t),2):
        ii_y = ii//2%5
        ii_x = ii//2//5
        print(ii, ii_x, ii_y)
        idx_sample = np.random.randint(0, seq_t[0].shape[0], size=1)[0]
        axes[ii_x][ii_y].imshow(seq_t[ii][idx_sample], cmap='Greys', vmin=-1, vmax=1)
        axes[ii_x][ii_y].set_title("Time="+str(ii), fontdict={"size":32})
        axes[ii_x][ii_y].tick_params(axis='both', which='major', labelsize=32)
        axes[ii_x][ii_y].set_xlabel("")
        axes[ii_x][ii_y].set_ylabel("")
        axes[ii_x][ii_y].set_xticks([])
        axes[ii_x][ii_y].set_yticks([])
    fig.tight_layout()
    plt.savefig("seq_t_T%.2f.png"%T, bbox_inches="tight")

    e_time = time.time()
    print("Time for plotting:: ", e_time-s_time)

plot_seq(T1)