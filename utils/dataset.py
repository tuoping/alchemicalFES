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
        if args.subset_size is not None:
            if isinstance( args.dataset_dir, list):
                all_data = None
                for d in args.dataset_dir:
                    if all_data is None:
                        all_data = np.load("%s/buffer.npy"%(d))
                    else:
                        all_data = np.concatenate([all_data, np.load("%s/buffer.npy"%(d))], axis=0)
            else:
                all_data = np.load("%s/buffer.npy"%(args.dataset_dir))
            np.random.shuffle(all_data)
            all_data = torch.from_numpy(all_data[:args.subset_size])
        else:
            if isinstance( args.dataset_dir, list):
                all_data = None
                for d in args.dataset_dir:
                    if all_data is None:
                        all_data = np.load("%s/buffer.npy"%(d))
                    else:
                        all_data = np.concatenate([all_data, np.load("%s/buffer.npy"%(d))], axis=0)
                all_data = torch.from_numpy(all_data)
            else:
                all_data = torch.from_numpy(np.load("%s/buffer.npy"%(args.dataset_dir)))

        print("loaded ", all_data.shape, all_data.dtype, args.dataset_dir)
        self.num_cls = 2
        self.seq_len = args.toy_seq_len
        self.alphabet_size = args.toy_simplex_dim

        self.seqs = all_data.reshape(-1,*args.toy_seq_dim).to(device=device, dtype=torch.int64)
        self.seqs[torch.where(self.seqs == -1)] = 0
        # self.clss = torch.full_like(self.seqs, 0).to(device="cuda", dtype=torch.int64)
        # self.clss = self.seqs.clone()
        if self.seq_len == 36:
            self.energy = ising_boltzman_prob(self.seqs)
            assert not torch.isinf(torch.exp((-self.energy+self.energy.min())/3.2)).any(), "max(reduced energy)=%f"(((-self.energy+self.energy.min())/3.2).max())
            assert (self.energy % 4 == 0).all()
            assert self.energy.max() <= 52
            self.energy_op = ((self.energy+72)//4).to(torch.int64)

        
        self.magn_cls = RC(torch.nn.functional.one_hot(self.seqs.reshape(-1), num_classes=self.alphabet_size).reshape(-1,*args.toy_seq_dim, self.alphabet_size), device=device).to(device=device).reshape(-1)
        self.magn_cls = (self.magn_cls // args.scale_magn).to(torch.int64)
        self.magn_cls = self.magn_cls


    def read_target_class(self, dataset_file, seq_L, scale_magn, subset_size):
        print("WARNNING: using target dataset", dataset_file)
        all_data = np.load(dataset_file)
        np.random.shuffle(all_data)
        all_data = torch.from_numpy(all_data[:subset_size])
        toy_seq_dim = (seq_L, seq_L)

        target_seqs = all_data.reshape(-1,*toy_seq_dim).to(device=self.seqs.device, dtype=torch.int64)
        target_seqs[torch.where(target_seqs == -1)] = 0

        self.magn_cls = RC(torch.nn.functional.one_hot(target_seqs.reshape(-1), num_classes=self.alphabet_size).reshape(-1,*toy_seq_dim, self.alphabet_size), device=self.seqs.device).to(device=self.seqs.device).reshape(-1)
        self.magn_cls = (self.magn_cls // scale_magn).to(torch.int64)
        self.magn_cls = self.magn_cls

        if seq_L != 6:
            raise Exception("Lattice size of the target class = %d, check if this is what you want."%seq_L)
        self.energy = ising_boltzman_prob(target_seqs)
        assert not torch.isinf(torch.exp((-self.energy+self.energy.min())/3.2)).any(), "max(reduced energy)=%f"(((-self.energy+self.energy.min())/3.2).max())
        assert (self.energy % 4 == 0).all()
        assert self.energy.max() <= 52
        self.energy_op = ((self.energy+72)//4).to(torch.int64)
            


    def make_custom_target_class(self):
        print("WARNNING: using custom FM target dataset")
        num_seq = self.seqs.shape[0]
        ### Data ensemble for analytical conditional probability
        ### Toy setting 1: noisy
        # self.data_class1 = torch.stack([torch.from_numpy(np.random.choice(np.arange(self.alphabet_size), size=args.toy_seq_len, replace=True)) for _ in range(num_seq)]).to(device)
        # self.data_class2 = torch.stack([torch.from_numpy(np.random.choice(np.arange(self.alphabet_size), size=args.toy_seq_len, replace=True)) for _ in range(num_seq)]).to(device)

        ### Toy setting 3: all spin down/noisy
        # probabilities = [0.9, 0.1]
        # self.data_class1 = torch.stack([torch.from_numpy(np.random.choice(np.arange(self.alphabet_size), size=args.toy_seq_len, replace=True, p=probabilities)) for _ in range(num_seq)]).to(device)
        
        ### Custom magn_guidance
        magn_cls_1 = torch.stack([torch.from_numpy(np.zeros(36*2)).to(torch.int64) for _ in range(num_seq//2)]).to(self.seqs.device).sum(-1)
        magn_cls_2 = torch.stack([torch.from_numpy(np.ones(36*2)).to(torch.int64) for _ in range(num_seq - num_seq//2)]).to(self.seqs.device).sum(-1)
        self.magn_cls = torch.cat([magn_cls_1, magn_cls_2], dim=0)
        self.magn_cls = self.magn_cls[torch.randperm(len(self.magn_cls))]

        ### Custom energy_guidance
        self.energy = -72.*torch.ones(num_seq)
        self.energy_op = torch.zeros(num_seq).to(torch.int64)
        pass

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.magn_cls[idx], self.energy[idx], self.energy_op[idx]
    


class AlCuDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()
        all_data = torch.from_numpy(np.load(f"data/{args.dataset_dir}/buffer_atypes.npy").reshape(-1,args.toy_simplex_dim,args.toy_seq_len))
        print("loaded ", all_data.shape)
        self.num_cls = 1
        self.seq_len = args.toy_seq_len
        self.alphabet_size = args.toy_simplex_dim

        if args.cls_ckpt is not None:
            distribution_dict = torch.load(os.path.join(os.path.dirname(args.cls_ckpt), 'toy_distribution_dict.pt'))
            self.probs = distribution_dict['probs']
            self.class_probs = distribution_dict['class_probs']
        else:
            # self.seqs_T0 = torch.softmax(torch.swapaxes(all_data, 1, 2), dim=2)
            # if args.dataset_scaleTemp:
            #     print("Rescaling dataset from 620K to 420K")
            #     self.seqs = torch.pow(self.seqs_T0, 620.0/420.0)
            # else:
            #     self.seqs = self.seqs_T0
    
            self.seqs = torch.argmax(torch.swapaxes(all_data, 1, 2), dim=2).reshape(-1, *args.toy_seq_dim)
            # self.clss = torch.argmax(torch.swapaxes(all_data, 1, 2), dim=2)
            self.clss = torch.full_like(self.seqs, 0)

            # from sklearn.cluster import KMeans
            # est2 = KMeans(n_clusters=2)
            # est2.fit(self.clss)
            # counts_labels = Counter(est2.labels_)
            # counts = torch.tensor([counts_labels[k] for k in [0,1]]).reshape(self.num_cls, 1,-1)
            # self.probs = counts / counts.sum(dim=-1, keepdim=True)
            # print("probs = ", self.probs)

            # self.class_probs = torch.softmax(torch.swapaxes(all_data, 1, 2), dim=2)

            # distribution_dict = {'probs': self.probs, 'class_probs': self.class_probs}
        # torch.save(distribution_dict, os.path.join(os.environ["MODEL_DIR"], 'toy_distribution_dict.pt' ))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.clss[idx]
