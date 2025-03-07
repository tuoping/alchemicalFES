import copy
import pickle

import torch, esm, random, os, json
import numpy as np
from Bio import SeqIO
from collections import Counter


def pbc(i,L=16):
    assert i>=-1 and i<=L
    if i-L == 0:
        return 0
    elif i == -1:
        return L-1
    else:
        return i
    

def ising_boltzman_prob(seq, J=1):
    shape = seq.shape
    spins = seq.clone()
    spins[torch.where(spins==0)]=-1
    B,H,W = shape
    E = torch.zeros(B, device=spins.device)
    for i in range(H):
        for j in range(W):
            E += -spins[:,i,j]*spins[:,pbc(i-1,L=H),j]*J
            E += -spins[:,i,j]*spins[:,pbc(i+1,L=H),j]*J
            E += -spins[:,i,j]*spins[:,i,pbc(j-1,L=H)]*J
            E += -spins[:,i,j]*spins[:,i,pbc(j+1,L=H)]*J

    E /= 2
    return E


def RC(logits, device="cuda"):
    assert logits.shape[-1] == 2
    B,H,W,K = logits.shape
    RC = torch.sum(logits*torch.tensor([-1,1]).to(device=device)[None,None,None,:], dim=-1)
    RC = torch.sum(RC.reshape(B, -1), dim=-1)
    RC = H*W-RC
    return RC.reshape(-1,1)

class IsingDataset(torch.utils.data.Dataset):
    def __init__(self, args, device="cuda"):
        super().__init__()

        if isinstance( args.dataset_dir, list):
            all_data = [np.load("%s"%(d)) for d in args.dataset_dir]
            all_data = np.concatenate(all_data)
        else:
            all_data = np.load("%s"%(args.dataset_dir))
        np.random.shuffle(all_data)
        if args.subset_size is not None:
            all_data = torch.from_numpy(all_data[:args.subset_size])
        else:
            all_data = torch.from_numpy(all_data)

        print("loaded ", all_data.shape, all_data.dtype, args.dataset_dir)
        self.num_cls = 2
        self.seq_len = args.toy_seq_len
        self.alphabet_size = args.toy_simplex_dim

        self.seqs = all_data.reshape(-1,*args.toy_seq_dim).to(device=device, dtype=torch.int64)
        self.seqs[torch.where(self.seqs == -1)] = 0
        # self.clss = torch.full_like(self.seqs, 0).to(device="cuda", dtype=torch.int64)
        # self.clss = self.seqs.clone()
        # if self.seq_len == 36:
        self.energy = ising_boltzman_prob(self.seqs)
        assert not torch.isinf(torch.exp((-self.energy+self.energy.min())/3.2)).any(), "max(reduced energy)=%f"(((-self.energy+self.energy.min())/3.2).max())
        assert (self.energy % 4 == 0).all()
        assert self.energy.max()//args.scale_magn <= 52
        self.energy_op = ((self.energy//args.scale_magn+72)//4).to(torch.int64)

        
        self.magn_cls = RC(torch.nn.functional.one_hot(self.seqs.reshape(-1), num_classes=self.alphabet_size).reshape(-1,*args.toy_seq_dim, self.alphabet_size), device=device).to(device=device).reshape(-1)
        self.magn_cls = (self.magn_cls // args.scale_magn).to(torch.int64)


    def read_target_class(self, dataset_file, seq_L, scale_magn, subset_size):
        print("WARNNING: using target dataset", dataset_file)
        all_data = np.load(dataset_file)
        np.random.shuffle(all_data)
        all_data = torch.from_numpy(all_data[:subset_size])
        toy_seq_dim = (seq_L, seq_L)
        assert seq_L == all_data.shape[1] or (seq_L**2 == all_data.shape[1] and len(all_data.shape) == 2)

        target_seqs = all_data.reshape(-1,*toy_seq_dim).to(device=self.seqs.device, dtype=torch.int64)
        target_seqs[torch.where(target_seqs == -1)] = 0

        self.magn_cls = RC(torch.nn.functional.one_hot(target_seqs.reshape(-1), num_classes=self.alphabet_size).reshape(-1,*toy_seq_dim, self.alphabet_size), device=self.seqs.device).to(device=self.seqs.device).reshape(-1)
        self.magn_cls = (self.magn_cls // scale_magn).to(torch.int64)

        if seq_L != 6:
            print("WARNING:: Lattice size of the target class = %d, check if this is what you want."%seq_L)
        self.energy = ising_boltzman_prob(target_seqs)
        assert not torch.isinf(torch.exp((-self.energy+self.energy.min())/3.2)).any(), "max(reduced energy)=%f"(((-self.energy+self.energy.min())/3.2).max())
        print("Target energy between ", self.energy.min(), self.energy.max())
        self.energy = self.energy//scale_magn
        print("Scaled target energy between ", self.energy.min(), self.energy.max())
        print("Scaling factor = ", scale_magn)
        # assert (self.energy % 4 == 0).all()
        assert self.energy.max() <= 52
        self.energy_op = ((self.energy+72)//4).to(torch.int64)
            


    def make_custom_target_class(self):
        print("WARNNING: using custom target dataset: 6x6 Ising model with all ones or all minus ones")
        B,H,W = self.seqs.shape
        assert H == 6
        assert W == 6
    
        # Randomly decide which batches get +1 vs. -1
        # Here, sign_mask will be in {0, 1} shape [B].
        sign_mask = torch.randint(0, 2, (B,), device=self.seqs.device)
    
        # Convert sign_mask {0,1} → {+1, -1} in float
        # shape will be [B,1,1]; broadcast to [B,H,W]
        target_seqs = 2.0 * sign_mask.view(-1, 1, 1).float() - 1.0
        
        self.magn_cls = RC(torch.nn.functional.one_hot(target_seqs.reshape(-1), num_classes=self.alphabet_size).reshape(-1,*toy_seq_dim, self.alphabet_size), device=self.seqs.device).to(device=self.seqs.device).reshape(-1)
        self.energy = ising_boltzman_prob(target_seqs)
        assert not torch.isinf(torch.exp((-self.energy+self.energy.min())/3.2)).any(), "max(reduced energy)=%f"(((-self.energy+self.energy.min())/3.2).max())
        assert (self.energy % 4 == 0).all()
        assert self.energy.max() <= 52
        self.energy_op = ((self.energy+72)//4).to(torch.int64)


    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.magn_cls[idx], self.energy[idx], self.energy_op[idx]
    
        


class IsingDataset_mixT(torch.utils.data.Dataset):
    def __init__(self, args, device="cuda"):
        super().__init__()
        self.seqs_T = []
        self.energy_T = []
        self.energy_op_T = []
        self.magn_cls_T = []
        self.T = [3.2, 2.2]
        for T in self.T:
            all_data = torch.from_numpy(np.load("data/ising-latt6x6-T4.0/latt6x6/buffer-S%.2f.npy"%T)[:500000])


            print("loaded ", all_data.shape, all_data.dtype, args.dataset_dir)
            self.num_cls = 2
            self.seq_len = args.toy_seq_len
            self.alphabet_size = args.toy_simplex_dim
            
            seqs = all_data.reshape(-1,*args.toy_seq_dim).to(device=device, dtype=torch.int64)
            seqs[torch.where(seqs == -1)] = 0
            self.seqs_T.append(seqs)
            if self.seq_len == 36:
                energy = ising_boltzman_prob(seqs)
                self.energy_T.append(energy)
                assert not torch.isinf(torch.exp((-energy+energy.min())/3.2)).any(), "max(reduced energy)=%f"(((-energy+energy.min())/3.2).max())
                assert (energy % 4 == 0).all()
                assert energy.max() <= 52
                energy_op = ((energy+72)//4).to(torch.int64)
                self.energy_op_T.append(energy_op)
            
            
            magn_cls = RC(torch.nn.functional.one_hot(seqs.reshape(-1), num_classes=self.alphabet_size).reshape(-1,*args.toy_seq_dim, self.alphabet_size), device=device).to(device=device).reshape(-1)
            magn_cls = (magn_cls // args.scale_magn).to(torch.int64)
            self.magn_cls_T.append(magn_cls)


    def __len__(self):
        return len(self.seqs_T[0])

    def __getitem__(self, idx):
        ridxT = torch.randperm(len(self.T))[0]
        return self.seqs_T[ridxT][idx], self.magn_cls_T[ridxT][idx], self.energy_T[ridxT][idx], self.energy_op_T[ridxT][idx]
    
