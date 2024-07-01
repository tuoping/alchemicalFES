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

def sample_cond_prob_path_2d(hyperparams, seq, channels):
    shape = seq.shape
    batchsize = seq.shape[0]
    seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=channels).reshape(*shape, channels)
    t = torch.from_numpy(scipy.stats.expon().rvs(size=batchsize)*hyperparams.time_scale).to(seq.device).float()

    alphas = torch.ones(*shape, channels, device=seq.device)
    alphas = alphas + t[:, None, None, None]*seq_onehot
    xt = torch.distributions.Dirichlet(alphas).sample()
    return xt, t+1


def RC(logits):
    assert logits.shape[-1] == 2
    B = logits.shape[0]
    RC = torch.sum(logits*torch.tensor([-1,1], device=logits.device)[None,None,None,:], dim=-1)
    RC = torch.sum(RC.reshape(B, -1), dim=-1)
    return RC.reshape(-1,1)

    
class RCLoss():
    def __init__(self, RC):
        self.RC = RC
    
    def buffer2rc_trajs(self, x):
        self.rc_trajs = self.RC(x)
    
    def kde(self, x_grid, bandwidth=2., dump_hist=False):
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
        ### Data augmentation by flipping the binary choices
        if self.stage == "train":
            seq_symm = -seq+1
            seq = torch.cat([seq, seq_symm])
            cls = torch.cat([cls, cls])
            
        if self.hyperparams.model == "CNN3D":
            B, H, W, D = seq.shape
            xt, t = sample_cond_prob_path(self.hyperparams, seq, self.model.alphabet_size)
        elif self.hyperparams.model == "CNN2D":
            B, H, W = seq.shape
            xt, t = sample_cond_prob_path_2d(self.hyperparams, seq, self.model.alphabet_size)
        shape = seq.shape
        # self.plot_probability_path(t, xt)
        logits = self.model(xt, t, cls=None)
        if self.hyperparams.model == "CNN3D":
            logits = (logits.permute(0,2,3,4,1)).reshape(-1, self.model.alphabet_size)
        elif self.hyperparams.model == "CNN2D":
            logits = (logits.permute(0,2,3,1)).reshape(-1, self.model.alphabet_size)
        # logits.retain_grad()
        
        
        losses = self.hyperparams.prefactor_CE* torch.nn.functional.cross_entropy(logits, seq.reshape(-1), reduction='none').reshape(B,-1)
        self.lg("CELoss", losses)
        if self.hyperparams.mode is not None and "RC" in self.hyperparams.mode:
            xgrid = torch.linspace(-36, 36, 36+1, device=logits.device)
            seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=self.model.alphabet_size).reshape(*shape, self.model.alphabet_size)
            self.RCL.buffer2rc_trajs(seq_onehot)
            rc_seq = self.RCL.kde(xgrid, 1., dump_hist=True)

            norm_logits = torch.nn.functional.softmax(logits, dim=-1)
            self.RCL.buffer2rc_trajs(norm_logits)
            if self.stage == "val":
                rc_logits= self.RCL.kde(xgrid, 1., dump_hist=False)
                rc_loss = (1-rc_logits)**2*torch.nn.functional.kl_div(rc_logits.reshape(-1, *xgrid.shape), rc_seq.reshape(-1, *xgrid.shape), reduction='none', log_target=False).reshape(B, -1)
                np.save(os.path.join(os.environ["work_dir"], f"logits_train_step{self.trainer.global_step}"), norm_logits.cpu())
            else:
                rc_logits= self.RCL.kde(xgrid, 1.)
                if "focal" in self.hyperparams.mode:
                    rc_loss = (1-rc_logits)**2*torch.nn.functional.kl_div(rc_logits.reshape(-1, *xgrid.shape), rc_seq.reshape(-1, *xgrid.shape), reduction='none', log_target=False).reshape(B, -1)
                else:
                    rc_loss = (1-rc_logits)**2*torch.nn.functional.kl_div(rc_logits.reshape(-1, *xgrid.shape), rc_seq.reshape(-1, *xgrid.shape), reduction='none', log_target=False).reshape(B, -1)
            self.lg("RCLoss", rc_loss*self.hyperparams.prefactor_RC)
            # rc_loss.sum().backward()
            # print(logits.grad)
            losses += rc_loss.mean()*self.hyperparams.prefactor_RC
            if self.stage == "train":
                current_lr = self.optimizers().param_groups[0]['lr']
                self.lg("LR", torch.tensor([current_lr]))

        losses = losses.mean(-1)
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
        np.save(os.path.join(os.environ["work_dir"], f"logits_val_step{self.trainer.global_step}_inttime{1.00}"), xt.cpu().to(torch.float16))
        # return xt, x0
        t_span = torch.linspace(1, self.hyperparams.alpha_max, self.hyperparams.num_integration_steps, device = self.device)
        for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            logits = self.model(xt, t=s[None].expand(B))
            flow_probs = torch.nn.functional.softmax(logits.permute(0,2,3,1)/self.hyperparams.flow_temp, -1)
            default_dtype = flow_probs.dtype
            if not torch.allclose((flow_probs.reshape(B,-1,K)).sum(2), torch.ones((B, H*W), device=self.device, dtype=default_dtype), atol=1e-4) or not (flow_probs >= 0).all():
                print(f'WARNING: flow_probs.min(): {flow_probs.min()}. Some values of flow_probs do not lie on the simplex. There are we are {(flow_probs<0).sum()} negative values in flow_probs of shape {flow_probs.shape} that are negative. We are projecting them onto the simplex.')
                flow_probs = simplex_proj(flow_probs.reshape(B,-1)).reshape(B,H,W,K)

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
            if not torch.allclose((xt.reshape(B,-1,K)).sum(2), torch.ones((B, H*W), device=self.device), atol=1e-4) or not (xt >= 0).all():
                print(f'WARNING: xt.min(): {xt.min()}. Some values of xt do not lie on the simplex. There are we are {(xt<0).sum()} negative values in xt of shape {xt.shape} that are negative. We are projecting them onto the simplex.')
                xt = simplex_proj(xt.reshape(B,-1)).reshape(B,H,W,K)
            np.save(os.path.join(os.environ["work_dir"], f"logits_val_step{self.trainer.global_step}_inttime{t}"), xt.cpu().to(torch.float16))
               
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
        scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
        # return optimizer
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # 调度器步进的步长：'epoch'或'step'
                'frequency': 1,  # 调度器步进的频率
            }
        }


    def plot_probability_path(self, t, xt):
        pass

    def lg(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        log = self._log
        log[self.stage + "_" + key].extend(data)
