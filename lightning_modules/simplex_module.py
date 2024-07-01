import torch
# from matplotlib import pyplot as plt
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning_modules.general_module import GeneralModule
from utils.logging import get_logger
from sklearn.cluster import KMeans

from torch.optim.lr_scheduler import StepLR
import numpy as np
import os, copy

from torch import nn
import torch.nn.functional as F

from model.dna_models import CNNModel3D, CNNModel2D, MLPModel
from utils.flow_utils import DirichletConditionalFlow, simplex_proj
from collections import Counter

import scipy
def sample_cond_prob_path(hyperparams, seq, channels):
    shape = seq.shape
    batchsize = seq.shape[0]
    seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=channels).reshape(*shape, channels)
    t = torch.from_numpy(scipy.stats.expon().rvs(size=batchsize)*hyperparams.time_scale).to(seq.device).float()
    alphas = torch.ones(*shape, channels, device=seq.device)
    alphas = alphas + t[:, None, None, None, None]*seq_onehot
    xt = torch.distributions.Dirichlet(alphas).sample()
    return xt, t+1

def sample_cond_prob_path_2d(hyperparams, seq, seq_t, channels):
    shape = seq.shape
    batchsize = seq.shape[0]
    seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=channels).reshape(*shape, channels)
    t_increment = torch.rand(batchsize).to(seq.device).float()
    ### add t points between Inf kBT~the maximum kBT (0~min diffusion step)
    t_min = seq_t.min()
    t_increment = (t_increment-1.)*(t_min-0.)
    t = seq_t
    t[torch.where(seq_t == t_min)] += t_increment[torch.where(seq_t == t_min)]
    alphas = torch.ones(*shape, channels, device=seq.device)
    alphas = alphas + t[:, None, None, None]*seq_onehot
    xt = torch.distributions.Dirichlet(alphas).sample()
    return xt, t+1


def pbc(i,L):
    assert i>=-1 and i<=L
    if i-L == 0:
        return 0
    elif i == -1:
        return L-1
    else:
        return i

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
    


class simplexModule(GeneralModule):
    def __init__(self, channels, num_cls, hyperparams):
        super().__init__(hyperparams)
        self.load_model(channels, num_cls, hyperparams)
        self.condflow = DirichletConditionalFlow(K=self.model.alphabet_size, alpha_spacing=0.001, alpha_max=hyperparams.alpha_max)
        self.hyperparams = hyperparams
        self.RCL = RCLoss(RC)

    def load_model(self, channels, num_cls, hyperparams):
        if hyperparams.model == "CNN3D":
            self.model = CNNModel3D(hyperparams, channels, num_cls)
        elif hyperparams.model == "CNN2D":
            self.model = CNNModel2D(hyperparams, channels, num_cls, classifier=hyperparams.classifier)
        elif hyperparams.model == "MLP":
            self.model = MLPModel(hyperparams, channels, num_cls)
        else:
            raise Exception("Unrecognized model type")


    def validation_step(self, batch, batch_idx):
        self.stage = 'val'
        loss = self.general_step(batch, batch_idx)
        self.log('val_loss', torch.tensor(self._log["val_loss"]).mean(), prog_bar=True)


    def general_step(self, batch, batch_idx=None):
        seq, seq_t = batch
        ### Data augmentation by flipping the binary choices
        seq_symm = -seq+1
        seq = torch.cat([seq, seq_symm])
        seq_t = torch.cat([seq_t, seq_t])
        if self.stage == "val":
            np.save("seq.npy", seq.detach().cpu().numpy())
            np.save("seq_t.npy", seq_t.detach().cpu().numpy())
            
        if self.hyperparams.model == "CNN3D":
            B, H, W, D = seq.shape
            xt, t = sample_cond_prob_path(self.hyperparams, seq, self.model.alphabet_size)
        elif self.hyperparams.model == "CNN2D":
            B, H, W = seq.shape
            xt, t = sample_cond_prob_path_2d(self.hyperparams, seq, seq_t, self.model.alphabet_size)
            if self.stage == "val":
                np.save("t.npy", t.detach().cpu().numpy())

        shape = seq.shape
            
        # self.plot_probability_path(t, xt)
        if self.hyperparams.classifier:
            logits, energy = self.model(xt, t, cls=None)
        else:
            logits = self.model(xt, t, cls=None)

        if self.hyperparams.model == "CNN3D":
            logits = (logits.permute(0,2,3,4,1)).reshape(-1, self.model.alphabet_size)
        elif self.hyperparams.model == "CNN2D":
            logits = (logits.permute(0,2,3,1)).reshape(-1, self.model.alphabet_size)
        # logits.retain_grad()
        if self.hyperparams.mode is not None and "focal" in self.hyperparams.mode:
            raise Exception("Focal on CE loss leads to pathological behaviors")
        elif self.hyperparams.mode is not None and "t-dependent-focal" in self.hyperparams.mode:
            raise Exception("Focal on CE loss leads to pathological behaviors")
        else:
            losses = torch.nn.functional.cross_entropy(logits, seq.reshape(-1), reduction='none').reshape(B,-1)
            self.lg("CELoss", losses)
            losses = self.hyperparams.prefactor_CE*losses
            # losses.mean().backward()
            # np.save("Grad_celoss_logits.npy", logits.grad.detach().cpu().numpy())
            # raise RuntimeError
        losses = losses.mean(-1)
        '''
        if self.hyperparams.mode is not None and "Energy" in self.hyperparams.mode:
            norm_logits = torch.nn.functional.softmax(logits, dim=-1)
            E_pred = ising_boltzman_prob(norm_logits.reshape([*shape,self.model.alphabet_size]))
            eloss = ((E_pred*seq_t/self.hyperparams.alpha_min_kBT)*seq_prob).reshape(-1)
            self.lg("EnergyLoss", eloss)
            losses += self.hyperparams.prefactor_E* eloss
        '''
        if self.hyperparams.classifier:
            raise Exception("Multitask later!")
            # lossses_energy = self.hyperparams.prefactor_E* (energy.ravel()-seq_energy)**2
            # self.lg("Eloss", lossses_energy)
            # losses = losses+lossses_energy
        if self.hyperparams.mode is not None and "multinomial" in self.hyperparams.mode:
            seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=self.model.alphabet_size).reshape(*shape, self.model.alphabet_size).float()
            logsoftmax = torch.nn.LogSoftmax(dim=-1)
            multinomial_loglogits = torch.sum(logsoftmax(logits).reshape(B,H*W,self.model.alphabet_size), axis=1)
            multinomial_seq = torch.prod(seq_onehot.reshape(B,H*W,self.model.alphabet_size), axis=1 )
            multinomial_loss = -(multinomial_loglogits*multinomial_seq).reshape(B,-1)
            self.lg("MLoss", multinomial_loss)
            losses += multinomial_loss.mean(-1)*self.hyperparams.prefactor_M

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
        
        if self.stage == "train":
            current_lr = self.optimizers().param_groups[0]['lr']
            self.lg("LR", torch.tensor([current_lr]))

        
        self.lg("loss", losses)

        if self.stage == "val":
            if self.hyperparams.model == "CNN3D":
                logits_pred, _ = self.dirichlet_flow_inference(seq)
            elif self.hyperparams.model == "CNN2D":
                logits_pred, _ = self.dirichlet_flow_inference_2d(seq)
            seq_pred = torch.argmax(logits_pred, dim=-1)
            np.save(os.path.join(os.environ["work_dir"], f"seq_val_step{self.trainer.global_step}"), seq_pred.cpu())
            np.save(os.path.join(os.environ["work_dir"], f"logits_val_step{self.trainer.global_step}"), logits_pred.cpu())
        return losses.mean()
    
    def training_step(self, batch, batch_idx):
        self.stage = "train"
        opt = self.optimizers()
        # manually delete scheduler in the Trainer
        for param_group in opt.param_groups:
            param_group['lr'] = self.hyperparams.lr  # Set to the new learning rate
        loss = self.general_step(batch)
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.stage = "val"
        loss = self.general_step(batch, batch_idx)


    @torch.no_grad()
    def dirichlet_flow_inference_2d(self, seq):

        B, H, W = seq.shape
        K = self.model.alphabet_size
        ### Use biased prior to predict biased FES with the unbiased model
        ### !!! Doesn't work
        # if self.hyperparams.prior_from_file:
        #     data = torch.from_numpy(np.load("/nfs/scistore14/chenggrp/ptuo/NeuralRG/dirichlet-flow-matching-test3/data/ising-latt4x4-T4.0/latt4x4/probbias/kappa176inv/buffer-S88.00.npy")).reshape(-1,H*W)
        #     idx_sel = torch.randint(data.shape[0], (B,))
        #     data[torch.where(data == -1)] = 0
        #     ones_tensor = torch.ones(1)
        #     x0 = torch.nn.functional.one_hot(data[idx_sel].reshape(-1).to(device=seq.device, dtype=torch.int64), num_classes=K).reshape(B,H,W,K).to(dtype=ones_tensor.dtype)
        #     noise = torch.randn_like(x0[:,:,:,:])*0.05
        #     x0[:,:,:,:] = x0[:,:,:,:] + noise
        #     x0 = simplex_proj(x0.reshape(B,-1)).reshape(B,H,W,K)
        # else:
        x0 = torch.distributions.Dirichlet(torch.ones(B, H, W, K, device=seq.device)).sample()

        eye = torch.eye(K).to(x0)
        xt = x0.clone()
        np.save(os.path.join(os.environ["work_dir"], f"logits_val_inttime{1.00}"), xt.cpu().to(torch.float16))
        np.save(os.path.join(os.environ["work_dir"], f"xt_val_inttime{1.00}"), xt.cpu().to(torch.float16))
        # return xt, x0
        t_span = torch.linspace(1., self.hyperparams.alpha_max, self.hyperparams.num_integration_steps, device = self.device)
        for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            if self.hyperparams.classifier:
                logits,_ = self.model(xt, t=s[None].expand(B))
            else:
                logits = self.model(xt, t=s[None].expand(B))
            flow_probs = torch.nn.functional.softmax(logits.permute(0,2,3,1)/self.hyperparams.flow_temp, -1)
            default_dtype = flow_probs.dtype
            if not torch.allclose((flow_probs.reshape(B,-1,K)).sum(2), torch.ones((B, H*W), device=self.device, dtype=default_dtype), atol=1e-4) or not (flow_probs >= 0).all():
                print(f'WARNING@t={t}:  flow_probs.min(): {flow_probs.min()}. Some values of flow_probs do not lie on the simplex. There are we are {(flow_probs<0).sum()} negative values in flow_probs of shape {flow_probs.shape} that are negative. We are projecting them onto the simplex.')
                flow_probs = simplex_proj(flow_probs.reshape(B,-1)).reshape(B,H,W,K)

            c_factor = self.condflow.c_factor(xt.cpu().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(xt)


            # self.inf_counter += 1
            if torch.isnan(c_factor).any():
                print(f'WARNING@t={t} NAN cfactor after: xt.min(): {xt.min()}, flow_probs.min(): {flow_probs.min()}')
                if self.hyperparams.allow_nan_cfactor:
                    c_factor = torch.nan_to_num(c_factor)
                    # self.nan_inf_counter += 1
                else:
                    raise RuntimeError(f'NAN cfactor after: xt.min(): {xt.min()}, flow_probs.min(): {flow_probs.min()}')

            if not (flow_probs >= 0).all(): print(f'flow_probs.min(): {flow_probs.min()}')
            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            # V=U*P: flow = conditional_flow*probability_path
            flow = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1)

            xt = xt + flow * (t - s)
            if not torch.allclose((xt.reshape(B,-1,K)).sum(2), torch.ones((B, H*W), device=self.device), atol=1e-4) or not (xt >= 0).all():
                print(f'WARNING@t={t}: xt.min(): {xt.min()}. Some values of xt do not lie on the simplex. There are we are {(xt<0).sum()} negative values in xt of shape {xt.shape} that are negative. We are projecting them onto the simplex.')
                xt = simplex_proj(xt.reshape(B,-1)).reshape(B,H,W,K)
            np.save(os.path.join(os.environ["work_dir"], "xt_val_inttime%.2f"%t), xt.cpu())
            np.save(os.path.join(os.environ["work_dir"], "logits_val_inttime%.2f"%t), logits.permute(0,2,3,1).cpu())
               
        return xt, x0

    @torch.no_grad()
    def dirichlet_flow_inference(self, seq):

        B, H, W, D = seq.shape
        K = self.model.alphabet_size

        x0 = torch.distributions.Dirichlet(torch.ones(B, H, W, D, K, device=seq.device)).sample()
        eye = torch.eye(K).to(x0)
        xt = x0.clone()
        np.save(os.path.join(os.environ["work_dir"], f"logits_val_step{self.trainer.global_step}_inttime{1.00}"), xt.cpu())

        t_span = torch.linspace(1, self.hyperparams.alpha_max, self.hyperparams.num_integration_steps, device = self.device)
        for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            logits = self.model(xt, t=s[None].expand(B))
            flow_probs = torch.nn.functional.softmax(logits.permute(0,2,3,4,1)/self.hyperparams.flow_temp, -1)

            if not torch.allclose((flow_probs.reshape(B,-1,K)).sum(2), torch.ones((B, H*W*D), device=self.device), atol=1e-4) or not (flow_probs >= 0).all():
                print(f'WARNING: flow_probs.min(): {flow_probs.min()}. Some values of flow_probs do not lie on the simplex. There are we are {(flow_probs<0).sum()} negative values in flow_probs of shape {flow_probs.shape} that are negative. We are projecting them onto the simplex.')
                flow_probs = simplex_proj(flow_probs.reshape(B,-1)).reshape(B,H,W,D,K)

            c_factor = self.condflow.c_factor(xt.cpu().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(xt)


            # self.inf_counter += 1
            if torch.isnan(c_factor).any():
                print(f'NAN cfactor after: xt.min(): {xt.min()}, flow_probs.min(): {flow_probs.min()}')
                if self.hyperparams.allow_nan_cfactor:
                    c_factor = torch.nan_to_num(c_factor)
                    # self.nan_inf_counter += 1
                else:
                    raise RuntimeError(f'NAN cfactor after: xt.min(): {xt.min()}, flow_probs.min(): {flow_probs.min()}')

            if not (flow_probs >= 0).all(): print(f'flow_probs.min(): {flow_probs.min()}')
            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            # V=U*P: flow = conditional_flow*probability_path
            flow = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1)

            xt = xt + flow * (t - s)
            if not torch.allclose((xt.reshape(B,-1,K)).sum(2), torch.ones((B, H*W*D), device=self.device), atol=1e-4) or not (xt >= 0).all():
                print(f'WARNING: xt.min(): {xt.min()}. Some values of xt do not lie on the simplex. There are we are {(xt<0).sum()} negative values in xt of shape {xt.shape} that are negative. We are projecting them onto the simplex.')
                xt = simplex_proj(xt.reshape(B,-1)).reshape(B,H,W,D,K)
            np.save(os.path.join(os.environ["work_dir"], f"logits_val_step{self.trainer.global_step}_inttime{t}"), xt.cpu())
               
        return logits.permute(0,2,3,4,1), x0

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hyperparams.lr)
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
        return optimizer
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         'scheduler': scheduler,
        #         'interval': 'epoch',  # 调度器步进的步长：'epoch'或'step'
        #         'frequency': 1,  # 调度器步进的频率
        #     }
        # }

    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_closure, *args, **kwargs):
        # 手动梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.)
        optimizer.step(closure=optimizer_closure)


    def plot_probability_path(self, t, xt):
        pass

    def lg(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        log = self._log
        log[self.stage + "_" + key].extend(data)
