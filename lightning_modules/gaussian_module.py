import torch
# from matplotlib import pyplot as plt
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning_modules.general_module import GeneralModule
from utils.logging import get_logger
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

def sample_cond_vector_field_2d(hyperparams, seq, seq_t, channels):
    shape = seq.shape
    batchsize = seq.shape[0]
    seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=channels).reshape(*shape, channels)
    sample_x = torch.randn(size=seq_onehot.shape, device=seq.device)

    # t = 1 - (torch.from_numpy(scipy.stats.expon().rvs(size=batchsize)*hyperparams.time_scale).to(seq.device).float())
    t = seq_t

    sigma_t = 1-(1-hyperparams.sigma_min)*t
    sample_x *= sigma_t[:,None,None,None]
    sample_x += t[:,None,None,None]*seq_onehot

    ut = (seq_onehot - (1-hyperparams.sigma_min)*sample_x)/sigma_t[:,None,None,None]
    sample_x.requires_grad = False
    return sample_x, t, ut.float()


def ising_boltzman_prob(logits, J=1):
    assert logits.shape[-1] == 2
    B,H,W,K = logits.shape
    spins = torch.sum(logits*torch.tensor([-1,1], device=logits.device)[None,None,None,:], dim=-1)

    E = torch.zeros(B, device=logits.device)
    for i in range(H):
            E += -(spins[:,i,:]*spins[:,pbc(i-1,L=H),:]*J).sum(dim=-1)
            E += -(spins[:,i,:]*spins[:,pbc(i+1,L=H),:]*J).sum(dim=-1)
    for j in range(W):
            E += -(spins[:,:,j]*spins[:,:,pbc(j-1,L=H)]*J).sum(dim=1)
            E += -(spins[:,:,j]*spins[:,:,pbc(j+1,L=H)]*J).sum(dim=1)

    # for i in range(H):
    #     for j in range(W):
    #         E += -spins[:,i,j]*spins[:,pbc(i-1,L=H),j]*J
    #         E += -spins[:,i,j]*spins[:,pbc(i+1,L=H),j]*J
    #         E += -spins[:,i,j]*spins[:,i,pbc(j-1,L=H)]*J
    #         E += -spins[:,i,j]*spins[:,i,pbc(j+1,L=H)]*J

    E /= 2
    return E


def RC(logits):
    assert logits.shape[-1] == 2
    B = logits.shape[0]
    RC = torch.sum(logits*torch.tensor([-1,1], device=logits.device)[None,None,None,:], dim=-1)
    RC = torch.sum(RC.reshape(B, -1), dim=-1)
    return RC.reshape(-1)

    
class RCLoss():
    def __init__(self, RC):
        self.RC = RC
    
    def buffer2rc_trajs(self, x):
        self.rc_trajs = self.RC(x)

    
    def kde(self, x_grid, bandwidth=1., dump_hist=False):
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
        density = torch.zeros_like(x_grid, device=self.rc_trajs.device)

        # Compute KDE
        for i, x in enumerate(x_grid):
            # Kernel function (Gaussian kernel)
            kernel = torch.exp(-0.5 * ((self.rc_trajs - x) / bandwidth)**2) / torch.sqrt(2 * torch.tensor(3.141592653589793))
            # Sum over all data points
            density[i] = torch.sum(kernel) / (bandwidth * torch.sqrt(2 * torch.tensor(3.141592653589793)))

        norm_density = density/torch.sum(density)
        
        if dump_hist:
            F = -torch.log(norm_density+1e-19)
            path = os.path.join(
                os.environ["work_dir"], f"FES-RC.dat"
            )
            with open(path,"wb") as f:
                np.savetxt(f,x_grid.detach().cpu().numpy().reshape([1,-1]),fmt="%4.4e",delimiter=" ",header="RC")
                np.savetxt(f,F.detach().cpu().numpy().reshape([1,-1]),fmt="%4.4e",delimiter=" ",header="FES")
            path = os.path.join(
                os.environ["work_dir"], f"HIST-RC.dat"
            )
            with open(path,"wb") as f:
                np.savetxt(f,x_grid.detach().cpu().numpy().reshape([1,-1]),fmt="%4.4e",delimiter=" ",header="RC")
                np.savetxt(f,norm_density.detach().cpu().numpy().reshape([1,-1]),fmt="%4.4e",delimiter=" ",header="HIST")
        # return torch.sum(F, dim=0, keepdim=True)
        return norm_density

    def kde_t(self, t, x_grid, bandwidth=[2.0, 2.0], stage="train"):
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
        assert len(x_grid.shape) == 3
        assert x_grid.shape[2] == 2
        B = self.rc_trajs.shape[0]
        density = torch.zeros(*x_grid.shape[:-1], device=self.rc_trajs.device)
        # Compute KD
        kernel = torch.exp(-0.5*((self.rc_trajs[:,None,None]-x_grid[None,:,:,0])/bandwidth[0] )**2 -0.5*((t[:,None,None]-x_grid[None,:,:,1])/bandwidth[1] )**2 ) /2 *torch.pi
        density = torch.sum(kernel, dim=0)
        norm_density = density/density.sum(dim=0)[None,:]
        return norm_density
    
    


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
        seq, seq_t, seq_prob = batch
        ### Data augmentation by flipping the binary choices
        seq_symm = -seq+1
        seq = torch.cat([seq, seq_symm])
        seq_t = torch.cat([seq_t, seq_t])
        seq_prob = torch.cat([seq_prob, seq_prob])
        if self.stage == "val":
            np.save("seq.npy", seq.detach().cpu().numpy())
            np.save("seq_t.npy", seq_t.detach().cpu().numpy())


        if self.hyperparams.model == "CNN3D":
            B, H, W, D = seq.shape
            xt, t, ut = sample_cond_vector_field(self.hyperparams, seq, self.model.alphabet_size)
        elif self.hyperparams.model == "CNN2D":
            B, H, W = seq.shape
            xt, t, ut = sample_cond_vector_field_2d(self.hyperparams, seq, seq_t, self.model.alphabet_size)
        shape = seq.shape

        logits = self.model(xt, t, cls=None)

        
        ut_model = self.model(xt, t, cls=None)
        if self.hyperparams.model == "CNN3D":
            losses = torch.norm((ut_model.permute(0,2,3,4,1)).reshape(B, -1, self.model.alphabet_size) - ut.reshape(B, -1, self.model.alphabet_size), dim=-1)**2/2.
        elif self.hyperparams.model == "CNN2D":
            if self.hyperparams.mode is not None and "t-weighted" in self.hyperparams.mode:
                losses = t[:,None]*torch.norm((ut_model.permute(0,2,3,1)).reshape(B, -1, self.model.alphabet_size) - ut.reshape(B, -1, self.model.alphabet_size), dim=-1)**2/2.
            else:
                losses = torch.norm((ut_model.permute(0,2,3,1)).reshape(B, -1, self.model.alphabet_size) - ut.reshape(B, -1, self.model.alphabet_size), dim=-1)**2/2.
            # self.lg("FMloss", losses)

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
            np.save(os.path.join(os.environ["work_dir"], f"seq_val_step{self.trainer.global_step}"), seq_pred.cpu())
            np.save(os.path.join(os.environ["work_dir"], f"logits_val_step{self.trainer.global_step}"), logits_pred.cpu())
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
        xx = torch.normal(0, 1*self.hyperparams.time0_scale, size=(B,H,W,K), device=self.device)
        np.save(os.path.join(os.environ["work_dir"], f"logits_val_step{self.trainer.global_step}_inttime{0.0}"), xx.cpu())
        seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=K).reshape(B,H,W,K)
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
        for tt in t_span[1:]:
            samples_tt = torch.ones(B, device=self.device)*tt
            u_t = self.model(xx, samples_tt)
            xx = xx + u_t.permute(0,2,3,1)*1./self.hyperparams.num_integration_steps

            xx_t.append(xx)
            np.save(os.path.join(os.environ["work_dir"], f"logits_val_step{self.trainer.global_step}_inttime{tt}"), xx.cpu())
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
        self.load_model(channels, num_cls, hyperparams)
        self.hyperparams = hyperparams
        self.RCL = RCLoss(RC)

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
        seq, seq_t, seq_prob = batch
        ### Data augmentation by flipping the binary choices
        seq_symm = -seq+1
        seq = torch.cat([seq, seq_symm])
        seq_t = torch.cat([seq_t, seq_t])
        seq_prob = torch.cat([seq_prob, seq_prob])
        if self.stage == "val":
            np.save("seq.npy", seq.detach().cpu().numpy())
            np.save("seq_t.npy", seq_t.detach().cpu().numpy())

        if self.hyperparams.model == "CNN3D":
            B, H, W, D = seq.shape
            xt, t, ut = sample_cond_vector_field(self.hyperparams, seq, self.model.alphabet_size)
        elif self.hyperparams.model == "CNN2D":
            B, H, W = seq.shape
            xt, t, ut = sample_cond_vector_field_2d(self.hyperparams, seq, seq_t, self.model.alphabet_size)
        shape = seq.shape

        logits = self.model(xt, t, cls=None)
        if self.hyperparams.model == "CNN3D":
            logits = (logits.permute(0,2,3,4,1)).reshape(-1, self.model.alphabet_size)
        elif self.hyperparams.model == "CNN2D":
            logits = (logits.permute(0,2,3,1)).reshape(-1, self.model.alphabet_size)
        losses = torch.nn.functional.cross_entropy(logits, seq.reshape(-1), reduction='none').reshape(B,-1)
        self.lg("FMloss", losses)

        if self.hyperparams.mode is not None and "RC" in "".join(self.hyperparams.mode):
            xgrid = torch.linspace(-36, 36, 36+1, device=logits.device)
            seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=self.model.alphabet_size).reshape(*shape, self.model.alphabet_size).float()
            self.RCL.buffer2rc_trajs(seq_onehot)
            if "RC-t" in self.hyperparams.mode:
                tgrid = torch.linspace(0, self.hyperparams.alpha_min_kBT, (self.hyperparams.tgrid_num_alpha)).to(seq.device)
                _meshgrid = torch.meshgrid(xgrid, tgrid)
                meshgrid = torch.stack([_meshgrid[0].ravel(), _meshgrid[1].ravel()]).T
                meshgrid = meshgrid.reshape(37, self.hyperparams.tgrid_num_alpha, 2)
                rc_seq = self.RCL.kde_t(t, meshgrid, bandwidth=[2.0, self.hyperparams.tgrid_bandwidth], stage=self.stage+"_gt")
            elif "RC-distance":
                rc_seq = self.RCL.rc_trajs
            else:
                raise Exception("ERROR:: Wrong RC loss type")
            norm_logits = torch.nn.functional.softmax(logits, dim=-1)
            self.RCL.buffer2rc_trajs(norm_logits.reshape(B,H,W,self.model.alphabet_size))
            if "RC-t" in self.hyperparams.mode:
                rc_logits = self.RCL.kde_t(t, meshgrid, bandwidth=[2.0, self.hyperparams.tgrid_bandwidth])
            elif "RC-distance":
                rc_logits = self.RCL.rc_trajs
            else:
                raise Exception("ERROR:: Wrong RC loss type")
            if "RC-t" in self.hyperparams.mode:
                rc_logits = rc_logits.permute(1,0)
                rc_seq = rc_seq.permute(1,0)
                # rc_loss = torch.nn.functional.cross_entropy(rc_logits, rc_seq, reduction="none").reshape(1,-1)
                rc_loss = (-rc_seq*torch.log(rc_logits+1e-12)).reshape(1,-1)
                if self.stage == "val":
                    np.save("rc_logits.npy", rc_logits.detach().cpu().numpy())
                    np.save("rc_seq.npy", rc_seq.detach().cpu().numpy())
                    np.save("rc_loss.npy", rc_loss.detach().cpu().numpy())
                # rc_loss.mean().backward()
                # np.save("Grad_rcloss_logits.npy", logits.grad.detach().cpu().numpy())
                # raise RuntimeError
            elif "RC-distance":
                # rc_loss = torch.nn.functional.cross_entropy(rc_logits.reshape(-1, *xgrid.shape), rc_seq.reshape(-1, *xgrid.shape), reduction="none").reshape(B,-1)
                rc_loss = (rc_logits-rc_seq)**2
            else:
                raise Exception("ERROR:: Wrong RC loss type")
            self.lg("RCLoss", rc_loss*self.hyperparams.prefactor_RC)


            losses += rc_loss.mean(-1)*self.hyperparams.prefactor_RC
        


        losses = losses.mean(-1)
        self.lg("loss", losses)

        if self.stage == "val":
            if self.hyperparams.model == "CNN3D":
                logits_pred = self.gaussian_flow_inference(seq)
            elif self.hyperparams.model == "CNN2D":
                logits_pred = self.gaussian_flow_inference_2d(seq)

            seq_pred = torch.argmax(logits_pred, dim=-1)
            np.save(os.path.join(os.environ["work_dir"], f"seq_val_step{self.trainer.global_step}"), seq_pred.cpu())
            np.save(os.path.join(os.environ["work_dir"], f"logits_val_step{self.trainer.global_step}"), logits_pred.cpu())
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
        xx = torch.normal(0, 1*self.hyperparams.time0_scale, size=(B,H,W,K), device=self.device)
        np.save(os.path.join(os.environ["work_dir"], f"logits_val_step{self.trainer.global_step}_inttime{0.0}"), xx.cpu())
        
        seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=K).reshape(B,H,W,K)
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
            flow_probs = torch.nn.functional.softmax(logits.permute(0,2,3,1)/self.hyperparams.flow_temp, -1)
            sigma_t = 1-(1-self.hyperparams.sigma_min)*tt
            xx_1 = xx_t[0]*sigma_t
            xx_1 += tt*seq_onehot
    
            ut = (xx_1 - xx)/(tt-ss)
            xx = xx + flow_probs*ut*(tt-ss)

            xx_t.append(xx)
            np.save(os.path.join(os.environ["work_dir"], f"logits_val_step{self.trainer.global_step}_inttime{tt}"), xx.cpu())
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
