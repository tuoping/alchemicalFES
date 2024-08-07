import torch
# from matplotlib import pyplot as plt
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning_modules.general_module import GeneralModule
from utils.logging import get_logger
from utils.dataset import ising_boltzman_prob
from sklearn.cluster import KMeans

import numpy as np
import os, copy

from torch import nn
import torch.nn.functional as F

from model.dna_models import CNNModel3D, CNNModel2D, MLPModel

import scipy
def sample_cond_vector_field(hyperparams, seq, channels):
    shape = seq.shape
    batchsize = seq.shape[0]
    seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=channels).reshape(*shape, channels)
    sample_x = torch.randn(size=seq_onehot.shape, device=seq.device)
    
    t = 1 - (torch.from_numpy(scipy.stats.expon().rvs(size=batchsize)*hyperparams.time_scale).to(seq.device).float())
    sigma_t = 1-(1-hyperparams.sigma_min)*t
    sample_x *= sigma_t[:,None,None,None,None]
    sample_x += t[:,None,None,None,None]*seq_onehot

    ut = (seq_onehot - (1-hyperparams.sigma_min)*sample_x)/sigma_t[:,None,None,None,None]
    return sample_x, t, ut.float()

def sample_cond_vector_field_2d(hyperparams, seq, energies, channels):
    shape = seq.shape
    batchsize = seq.shape[0]
    
    t = torch.randn(batchsize, device=seq.device)**2*hyperparams.time_scale
    t.requires_grad_(True)

    # seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=channels).reshape(*shape, channels)
    # sample_x = torch.randn(size=seq_onehot.shape, device=seq.device)
    # sigma_t = 1-(1-hyperparams.sigma_min)*t
    # sample_x *= sigma_t[:,None,None,None]
    # sample_x += t[:,None,None,None]*seq_onehot
    # ut = (seq_onehot - (1-hyperparams.sigma_min)*sample_x)/sigma_t[:,None,None,None]

    sample_x = torch.nn.functional.softmax(-energies*(t)[:,None,None,None], dim=-1)
    dummy_grad_output = torch.ones_like(sample_x, device=seq.device)
    ut = torch.autograd.grad(outputs=[sample_x],
                               inputs=[t],
                               grad_outputs=[dummy_grad_output],
                               create_graph=True)[0]

    return sample_x, t, ut.float()


class Lattice:
    def __init__(self,L, d, BC='periodic'):
        self.L = L 
        self.d = d
        self.shape = [L]*d 
        self.Nsite = L**d 
        self.BC = BC

    def move(self, idx, d, shift):
        coord = self.index2coord(idx)
        coord[d] += shift

        if self.BC != 'periodic':
            if (coord[d]>=self.L) or (coord[d]<0):
                return None
        #wrap around because of the PBC
        if (coord[d]>=self.L): coord[d] -= self.L; 
        if (coord[d]<0): coord[d] += self.L; 

        return self.coord2index(coord)

    def index2coord(self, idx):
        coord = zeros(self.d, int) 
        for d in range(self.d):
            coord[self.d-d-1] = idx%self.L;
            idx /= self.L
        return coord 

    def coord2index(self, coord):
        idx = coord[0]
        for d in range(1, self.d):
            idx *= self.L; 
            idx += coord[d]
        return idx 

class Hypercube(Lattice):
    def __init__(self,L, d, BC='periodic'):
        super(Hypercube, self).__init__(L, d, BC)
        self.Adj = zeros((self.Nsite,self.Nsite), int)
        for i in range(self.Nsite):
            for d in range(self.d):
                j = self.move(i, d, 1)

                if j is not None:
                    self.Adj[i, j] = 1.0
                    self.Adj[j, i] = 1.0

from scipy.linalg import eigh, inv, det 
import math
class Ising:
    def __init__(self,L,d,T,name = None):
        if name is None:
            name = "Ising_l"+str(L)+"_d" +str(d)+"_t"+str(T)
        self.beta = 1.0
        self.lattice = Hypercube(L, d, 'periodic')
        self.K = self.lattice.Adj/T
    
        w, v = eigh(self.K)    
        offset = 0.1-w.min()
        self.K += np.eye(w.size)*offset
        sign, logdet = np.linalg.slogdet(self.K)
        #print (sign)
        #print (0.5*self.nvars[0] *(np.log(4.)-offset - np.log(2.*np.pi)) - 0.5*logdet)
        Kinv = torch.from_numpy(inv(self.K)).to(torch.float32)
        self.nvars = [L**d]

    def energy(self,x):
        return -(-0.5*(torch.mm(x.reshape(-1, self.nvars[0]),self.Kinv) * x.reshape(-1, self.nvars[0])).sum(dim=1) \
        + (torch.nn.Softplus()(2.*self.beta*x.reshape(-1, self.nvars[0])) - self.beta*x.reshape(-1, self.nvars[0]) - math.log(2.)).sum(dim=1))


class gaussianModule(GeneralModule):
    def __init__(self, channels, num_cls, hyperparams):
        super().__init__(hyperparams)
        self.load_model(channels, num_cls, hyperparams)
        self.hyperparams = hyperparams

    def load_model(self, channels, num_cls, hyperparams):
        if hyperparams.model == "CNN3D":
            self.model = CNNModel3D(hyperparams, channels, num_cls)
        elif hyperparams.model == "CNN2D":
            self.model = CNNModel2D(hyperparams, channels, num_cls)
        elif hyperparams.model == "MLP":
            self.model = MLPModel(hyperparams, channels, num_cls)
        else:
            raise Exception("Unrecognized model type")


    def validation_step(self, batch, batch_idx):
        self.stage = 'val'
        loss = self.general_step(batch, batch_idx)
        self.log('val_loss', torch.tensor(self._log["val_loss"]).mean(), prog_bar=True)

    def general_step(self, batch, batch_idx=None):
        seq, energies = batch
        # if self.stage == "train":
        #     seq_symm = -seq+1
        #     seq = torch.cat([seq, seq_symm])
        #     cls = torch.cat([cls, cls])
        B = seq.shape[0]
        if self.hyperparams.model == "CNN3D":
            xt, t, ut = sample_cond_vector_field(self.hyperparams, seq, self.model.alphabet_size)
        elif self.hyperparams.model == "CNN2D":
            xt, t, ut = sample_cond_vector_field_2d(self.hyperparams, seq, energies, self.model.alphabet_size)

        
        ut_model = self.model(xt, t, cls=None)
        if self.hyperparams.model == "CNN3D":
            losses = torch.norm((ut_model.permute(0,2,3,4,1)).reshape(B, -1, self.model.alphabet_size) - ut.reshape(B, -1, self.model.alphabet_size), dim=-1)**2/2.
        elif self.hyperparams.model == "CNN2D":
            losses = torch.norm((ut_model.permute(0,2,3,1)).reshape(B, -1, self.model.alphabet_size) - ut.reshape(B, -1, self.model.alphabet_size), dim=-1)**2/2.
            if self.stage == "val":
                np.save("t.npy", t.detach().cpu().numpy())
                np.save("ut.npy", ut.detach().cpu().numpy())
                np.save("losses.npy", losses.detach().cpu().numpy())

        if self.hyperparams.mode == "focal":
            norm_xt = torch.nn.functional.softmax(xt, dim=-1)
            fl = ((torch.pow(norm_xt, self.hyperparams.gamma_focal).sum(-1)).reshape(B, -1))*torch.norm((ut_model.permute(0,2,3,1)).reshape(B, -1, self.model.alphabet_size) - ut.reshape(B, -1, self.model.alphabet_size), dim=-1)**2/2.
            losses += fl

        losses = losses.mean(-1)
        self.lg("loss", losses)

        if self.stage == "val":
            if self.hyperparams.model == "CNN3D":
                logits_pred = self.gaussian_flow_inference(seq)
            elif self.hyperparams.model == "CNN2D":
                logits_pred = self.gaussian_flow_inference_2d(seq)

            seq_pred = torch.argmax(logits_pred, dim=-1)
            np.save(os.path.join(os.environ["work_dir"], "seq_val"), seq_pred.cpu())
            np.save(os.path.join(os.environ["work_dir"], "logits_val"), logits_pred.cpu())
        return losses.mean()
    
    def training_step(self, batch, batch_idx):
        self.stage = "train"
        loss = self.general_step(batch)
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.stage = "val"
        seq, energies = batch
        
        if self.hyperparams.model == "CNN3D":
            logits_pred = self.gaussian_flow_inference(seq)
        elif self.hyperparams.model == "CNN2D":
            with torch.enable_grad():
                logits_pred = self.gaussian_flow_inference_2d(seq)

        seq_pred = torch.argmax(logits_pred, dim=-1)
        np.save(os.path.join(os.environ["work_dir"], "seq_val"), seq_pred.cpu())
        np.save(os.path.join(os.environ["work_dir"], "logits_val"), logits_pred.cpu())

    @torch.enable_grad()
    def gaussian_flow_inference_2d(self, seq):
        B, H, W = seq.shape
        K = self.model.alphabet_size
        import matplotlib.pyplot as plt
        import seaborn as sns
        xx = torch.normal(-1, 1, size=(B,H,W), device=self.device) + torch.normal(1, 1, size=(B,H,W), device=self.device) 
        plt.figure()
        sns.histplot(xx)
        plt.savefig("xx0.png", bbox_inches="tight")
        # xx = torch.nn.functional.softmax(xx, dim=-1)
        np.save(os.path.join(os.environ["work_dir"], "logits_val_inttime%.2f"%0.0), xx.cpu())

        xx_t = []
        xx_t.append(xx)

        t_span = torch.linspace(0, 1, self.hyperparams.num_integration_steps, device = self.device)
        # for i, (ss, tt) in enumerate(zip(t_span[:-1], t_span[1:])):
        #     samples_ss = torch.ones(B, device=self.device)*ss
        # 
        #     logits = self.model(xx, samples_ss)
        #     flow_probs = torch.nn.functional.softmax(logits.permute(0,2,3,1), -1)
        #     sigma_t = 1-(1-self.hyperparams.sigma_min)*tt
        #     xx_1 = xx_t[0]*sigma_t
        #     xx_1 += tt*seq_onehot
        # 
        #     ut = (xx_1 - xx)/(tt-ss)
        #     xx = xx + flow_probs*ut*(tt-ss)
        ising_source = Ising(4,2,1)
        eps = 0.0
        for tt in t_span[:-1]:
            # samples_tt = torch.ones_like(seq, device=self.device)*tt
            # samples_tt.requires_grad_(True)
            # samples_tt = samples_tt.reshape(-1)

            # seq_t = xx.argmax(dim=-1).float()
            # seq_t.requires_grad_()
            # _energies = ising_boltzman_prob(seq_t)
            _energies = ising_source.energy(xx)
            u_t = _energies + _energies.mean(-1)
            
            xx = xx - (u_t*1./self.hyperparams.num_integration_steps).reshape(B,H,W,K) + np.sqrt(eps)*1.*torch.rand(xx.shape, device=seq.device)
            # u_t = self.model(xx, samples_tt)
            # xx = xx + u_t.permute(0,2,3,1)*1./self.hyperparams.num_integration_steps
            xx_t.append(xx)
            plt.figure()
            sns.histplot(xx)
            plt.savefig("xx%.4f.png"%tt, bbox_inches="tight")

            np.save(os.path.join(os.environ["work_dir"], "logits_val_inttime%.2f"%(tt)), xx.detach().cpu().numpy())
            np.save(os.path.join(os.environ["work_dir"], "ut_val_inttime%.2f"%(tt)), u_t.detach().cpu().numpy())
        return xx_t[-1]

    @torch.no_grad()
    def gaussian_flow_inference(self, seq):
        B, H, W, D = seq.shape
        K = self.model.alphabet_size
        xx = torch.normal(0, 1*self.hyperparams.time0_scale, size=(B,H,W,D,K), device=self.device)
        xx_t = []
        xx_t.append(xx)

        t_span = torch.linspace(0, 1, self.hyperparams.num_integration_steps, device = self.device)
        for tt in t_span:
            samples_tt = torch.ones(B, device=self.device)*tt
            u_t = self.model(xx, samples_tt)
            xx = xx + u_t.permute(0,2,3,4,1)*1./self.hyperparams.num_integration_steps
            xx_t.append(xx)
        return xx_t[-1].permute(0,4,1,2,3)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hyperparams.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
        return optimizer

    def plot_probability_path(self, t, xt):
        pass

    def lg(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        log = self._log
        log[self.stage + "_" + key].extend(data)



class gaussianModule_celoss(GeneralModule):
    def __init__(self, channels, num_cls, hyperparams):
        super().__init__(hyperparams)
        raise Exception("ERROR:: Training by cross entropy loss of the diffusion model using Gaussian basis is not implemented")
        self.load_model(channels, num_cls, hyperparams)
        self.hyperparams = hyperparams

    def load_model(self, channels, num_cls, hyperparams):
        if hyperparams.model == "CNN3D":
            self.model = CNNModel3D(hyperparams, channels, num_cls)
        elif hyperparams.model == "CNN2D":
            self.model = CNNModel2D(hyperparams, channels, num_cls)
        elif hyperparams.model == "MLP":
            self.model = MLPModel(hyperparams, channels, num_cls)
        else:
            raise Exception("Unrecognized model type")


    def validation_step(self, batch, batch_idx):
        self.stage = 'val'
        loss = self.general_step(batch, batch_idx)
        self.log('val_loss', torch.tensor(self._log["val_loss"]).mean(), prog_bar=True)

    def general_step(self, batch, batch_idx=None):
        seq, cls = batch
        seq_symm = -seq+1
        seq = torch.cat([seq, seq_symm])
        B = seq.shape[0]
        if self.hyperparams.model == "CNN3D":
            xt, t, ut = sample_cond_vector_field(self.hyperparams, seq, self.model.alphabet_size)
        elif self.hyperparams.model == "CNN2D":
            xt, t, ut = sample_cond_vector_field_2d(self.hyperparams, seq, self.model.alphabet_size)

        logits = self.model(xt, t, cls=cls)
        if self.hyperparams.model == "CNN3D":
            logits = (logits.permute(0,2,3,4,1)).reshape(-1, self.model.alphabet_size)
        elif self.hyperparams.model == "CNN2D":
            logits = (logits.permute(0,2,3,1)).reshape(-1, self.model.alphabet_size)
        losses = torch.nn.functional.cross_entropy(logits, seq.reshape(-1), reduction='none').reshape(B,-1)
        


        losses = losses.mean(-1)
        self.lg("loss", losses)

        if self.stage == "val":
            if self.hyperparams.model == "CNN3D":
                logits_pred = self.gaussian_flow_inference(seq)
            elif self.hyperparams.model == "CNN2D":
                logits_pred = self.gaussian_flow_inference_2d(seq)

            seq_pred = torch.argmax(logits_pred, dim=-1)
            np.save(os.path.join(os.environ["work_dir"], "seq_val"), seq_pred.cpu())
            np.save(os.path.join(os.environ["work_dir"], "logits_val"), logits_pred.cpu())
        return losses.mean()
    
    def training_step(self, batch, batch_idx):
        self.stage = "train"
        loss = self.general_step(batch)
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.stage = "val"
        loss = self.general_step(batch, batch_idx)


    @torch.no_grad()
    def gaussian_flow_inference_2d(self, seq):
        B, H, W = seq.shape
        K = self.model.alphabet_size
        xx = torch.normal(0, 1, size=(B,H,W,K), device=self.device)
        np.save(os.path.join(os.environ["work_dir"], "logits_val_inttime%.2f"%0.0), xx.cpu())
        
        # seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=K).reshape(B,H,W,K)
        xx_t = []
        xx_t.append(xx)
        # return xx_t[-1].permute(0,3,1,2)
        t_span = torch.linspace(0, 1, self.hyperparams.num_integration_steps, device = self.device)
        # for tt in t_span:
        #     samples_tt = torch.ones(B, device=self.device)*tt
        #     u_t = self.model(xx, samples_tt)
        #     xx = xx + u_t.permute(0,2,3,1)*1./self.hyperparams.num_integration_steps
        for i, (ss, tt) in enumerate(zip(t_span[:-1], t_span[1:])):
            samples_ss = torch.ones(B, device=self.device)*ss

            logits = self.model(xx, samples_ss)
            xx = torch.nn.functional.softmax(logits.permute(0,2,3,1), -1)
            xx_t.append(xx)
            np.save(os.path.join(os.environ["work_dir"], "logits_val_inttime%.2f"%tt), xx.cpu())
        return xx_t[-1]

    @torch.no_grad()
    def gaussian_flow_inference(self, seq):
        B, H, W, D = seq.shape
        K = self.model.alphabet_size
        xx = torch.normal(0, 1*self.hyperparams.time0_scale, size=(B,H,W,D,K), device=self.device)
        xx_t = []
        xx_t.append(xx)

        t_span = torch.linspace(0, 1, self.hyperparams.num_integration_steps, device = self.device)
        for tt in t_span:
            samples_tt = torch.ones(B, device=self.device)*tt
            u_t = self.model(xx, samples_tt)
            xx = xx + u_t.permute(0,2,3,4,1)*1./self.hyperparams.num_integration_steps
            xx_t.append(xx)
        return xx_t[-1].permute(0,4,1,2,3)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hyperparams.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
        return optimizer

    def plot_probability_path(self, t, xt):
        pass

    def lg(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        log = self._log
        log[self.stage + "_" + key].extend(data)
