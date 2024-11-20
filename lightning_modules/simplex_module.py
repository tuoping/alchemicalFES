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
from utils.flow_utils import DirichletConditionalFlow, simplex_proj, expand_simplex
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
    # t = torch.from_numpy(scipy.stats.expon().rvs(size=batchsize)*hyperparams.time_scale).to(seq.device).float()
    exponential_dist = torch.distributions.Exponential(1.0)
    t = exponential_dist.sample((batchsize,)).to(seq.device).float()*hyperparams.time_scale
    
    alphas = torch.ones(*shape, channels, device=seq.device)
    alphas = alphas + t[:, None, None, None]*seq_onehot
    xt = torch.distributions.Dirichlet(alphas).sample()
    return xt, t+1

class IsingGNN(nn.Module):
    def __init__(self, L):
        super(IsingGNN, self).__init__()
        self.L = L  # Lattice size
        self.edge_index = self.generate_lattice_edges(L)
    
    def generate_lattice_edges(self, L):
        edges = []
        for i in range(L):
            for j in range(L):
                # current node index
                node = i * L + j
                
                # Add edges to the right and down (to avoid double counting)
                right = i * L + (j + 1) % L
                down = ((i + 1) % L) * L + j
                left = i * L + (j - 1) % L
                up = ((i - 1) % L) * L + j
                
                edges.append([node, right])
                edges.append([node, down])
                edges.append([node, left])
                edges.append([node, up])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def forward_hard(self, x):
        B, seq_len, K = x.shape
        assert seq_len == self.L**2
        edge_index = self.edge_index.to(x.device)
        # Get the spin values for the edge pairs
        spin_values = torch.argmax(x, dim=-1) * 2 - 1  # Convert one-hot encoding to -1, 1
        total_energy = torch.zeros(B).to(x.device)
        for edge in range(edge_index.size(1)):
            source = edge_index[0, edge]
            target = edge_index[1, edge]
            total_energy += -spin_values[:,source] * spin_values[:,target]/2.
        return total_energy

    def forward_soft(self, x):
        B, seq_len, K = x.shape
        assert seq_len == self.L**2, "x.shape[1] != self.L**2 : %d != %d"%(seq_len, self.L**2)
        edge_index = self.edge_index.to(x.device)
        # Get the spin values for the edge pairs
        spin_values = torch.argmax(x, dim=-1) * 2 - 1  # Convert one-hot encoding to -1, 1
        total_energy = torch.zeros(B).to(x.device)
        for i in range(seq_len):
            # Calculate energy assuming the current spin is -1
            spin_values_neg1 = spin_values.clone()
            spin_values_neg1[:,i] = -1
            energy_neg1 = torch.zeros(B).to(x.device)
            for edge in range(edge_index.size(1)):
                source = edge_index[0, edge]
                target = edge_index[1, edge]
                energy_neg1 += -spin_values_neg1[:,source] * spin_values_neg1[:,target]/2.

            # Calculate energy assuming the current spin is 1
            spin_values_pos1 = spin_values_neg1
            spin_values_pos1[:,i] = 1
            energy_pos1 = torch.zeros(B).to(x.device)
            for edge in range(edge_index.size(1)):
                source = edge_index[0, edge]
                target = edge_index[1, edge]
                energy_pos1 += -spin_values_pos1[:,source] * spin_values_pos1[:,target]/2.
            del spin_values_pos1
            # Combine the energies weighted by the probabilities
            spin_energy = x[:, i, 0] * energy_neg1 + x[:, i, 1] * energy_pos1
            total_energy += spin_energy
        return total_energy/seq_len
    

def RC(logits):
    assert logits.shape[-1] == 2
    if len(logits.shape) == 4:
        B = logits.shape[0]
        RC = torch.sum(logits*torch.tensor([-1,1], device=logits.device)[None,None,None,:], dim=-1)
        RC = torch.sum(RC.reshape(B, -1), dim=-1)
    elif len(logits.shape) == 5:
        B = logits.shape[0]
        RC = torch.sum(logits*torch.tensor([-1,1], device=logits.device)[None,None,None,None,:], dim=-1)
        RC = torch.sum(RC.reshape(B, -1), dim=-1)
    else:
        print("logits dimension = ", logits.shape)
        raise Exception("logits dimension wrong")
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
    
    def kde_samples(self, x_grid, bandwidth=2., dump_hist=False):
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
        B = self.rc_trajs.shape[0]
        density = torch.zeros([B, *x_grid.shape], device=self.rc_trajs.device)

        # Compute KDE
        for i, x in enumerate(x_grid):
            # Kernel function (Gaussian kernel)
            kernel = torch.exp(-0.5 * ((self.rc_trajs - x) / bandwidth)**2) / torch.sqrt(2 * torch.tensor(3.141592653589793))
            # Sum over all data points
            density[:, i] = kernel[:, 0] / (bandwidth * torch.sqrt(2 * torch.tensor(3.141592653589793)))
        assert (torch.sum(density, dim=-1)>1e-3).all()
        norm_density = density/torch.sum(density, dim=-1)[:,None]
        
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

# Function to manually adjust the learning rate
def adjust_learning_rate(optimizer, epoch, last_epoch, initial_lr, decay_factor):
    new_lr = initial_lr * (decay_factor ** ((epoch - last_epoch)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

from copy import deepcopy
from utils.esm import upgrade_state_dict

class simplexModule(GeneralModule):
    def __init__(self, channels, num_cls=None, num_e=None, hyperparams=None, toy_data=None):
        super().__init__(hyperparams)
        self.load_model(channels, hyperparams, num_cls, num_e)
        self.condflow = DirichletConditionalFlow(K=self.model.alphabet_size, alpha_spacing=0.001, alpha_max=hyperparams.alpha_max)
        self.hyperparams = hyperparams
        self.RCL_seq = RCLoss(RC)
        self.RCL_logits = RCLoss(RC)
        self.ising_model = IsingGNN(hyperparams.seq_dim[0])
        self.loaded_uncond_model = False
        self.toy_data = toy_data

    def load_model(self, channels, hyperparams, num_cls=None, num_e=None):
        if hyperparams.model == "CNN3D":
            self.model = CNNModel3D(hyperparams, channels, num_cls, num_e)
        elif hyperparams.model == "CNN2D":
            self.model = CNNModel2D(hyperparams, channels, num_cls, num_e)
        elif hyperparams.model == "MLP":
            self.model = MLPModel(hyperparams, channels, num_cls)
        else:
            raise Exception("Unrecognized model type")

        self.log('val_loss', torch.tensor(self._log["val_loss"]).mean(), prog_bar=True)

    def general_step_3d(self, batch, batch_idx=None):
        seq, cls, energy, energy_op = batch
        ### Data augmentation by flipping the binary choices
        # if self.stage == "train":
        #     seq_symm = -seq+1
        #     seq = torch.cat([seq, seq_symm])
        #     probs = torch.cat([probs, probs])
            
        if self.hyperparams.model == "CNN3D":
            B, H, W, D = seq.shape
            xt, t = sample_cond_prob_path(self.hyperparams, seq, self.model.alphabet_size)
        elif self.hyperparams.model == "CNN2D":
            B, H, W = seq.shape
            xt, t = sample_cond_prob_path_2d(self.hyperparams, seq, self.model.alphabet_size)
        shape = seq.shape
        if self.hyperparams.cls_free_guidance:
            if self.hyperparams.guidance_op == "energy-magnetization":
                energy_op_inp = energy_op
                logits = self.model(xt, t, cls=cls, e=energy_op_inp)
            elif self.hyperparams.guidance_op == "magnetization":
                # cls_inp = torch.where(torch.rand(B, device=self.device) >= self.hyperparams.cls_free_noclass_ratio, cls.squeeze(), 73) # set fraction of the classes to the unconditional class
                logits = self.model(xt, t, cls=cls)
            else:
                raise Exception("Unrecognized guidance order parameter")
        else:
            logits = self.model(xt, t)
        if self.loaded_uncond_model:
            uncond_logits = self.uncond_model(xt,t)
            uncond_logits = uncond_logits.permute(0,2,3,4,1).reshape(-1,self.model.alphabet_size)

        logits = (logits.permute(0,2,3,4,1)).reshape(-1, self.model.alphabet_size)
        # logits.retain_grad()
        # CELoss = torch.nn.functional.cross_entropy(logits, seq.reshape(-1), reduction='none').reshape(B,-1)
        CELoss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(logits), seq.reshape(-1), reduction="none").reshape(B,-1)
        self.lg("CELoss", CELoss)
        if torch.isinf(torch.nn.functional.softmax(logits)).any():
            raise Exception("ERROR:: INF in: torch.nn.functional.softmax(logits)")

        losses = self.hyperparams.prefactor_CE* CELoss
        
        norm_logits = torch.nn.functional.softmax(logits)
        if self.loaded_uncond_model:
            norm_uncond_logits = torch.nn.functional.softmax(uncond_logits)
            logits_sum = norm_logits*norm_uncond_logits
            norm_logits_sum = torch.nn.functional.softmax(logits_sum)
        if self.hyperparams.mode is not None and "Energy" in self.hyperparams.mode:
            if not self.loaded_uncond_model:
                energy_pred = self.ising_model.forward_soft(norm_logits.reshape(B,H*W*D,self.model.alphabet_size))
            else:
                energy_pred = self.ising_model.forward_hard(norm_logits_sum.reshape(B,H*W*D,self.model.alphabet_size))
            assert not torch.isinf(energy_pred).any()
            EKLloss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(-energy_pred/self.hyperparams.EKLloss_temperature, dim=0), torch.nn.functional.log_softmax(-energy/self.hyperparams.EKLloss_temperature, dim=0), log_target=True, reduction="none")*B
            MSEloss = torch.abs(energy_pred-energy)
            # Eloss = (energy_pred - energy)**2
            self.lg("energy_klloss", EKLloss)
            self.lg("eneergy_mseloss", MSEloss)
            losses += MSEloss.mean()*self.hyperparams.prefactor_eloss_mse + EKLloss.mean()*self.hyperparams.prefactor_EKL
            ### DEBUG
            if self.stage == "val" and B <= 1024:
                # np.save(os.path.join(os.environ["work_dir"], "EKLloss"), EKLloss.detach().cpu().numpy())
                np.save(os.path.join(os.environ["work_dir"], "energy"), energy.detach().cpu().numpy())
                np.save(os.path.join(os.environ["work_dir"], "energy_pred"), energy_pred.detach().cpu().numpy())
                np.save(os.path.join(os.environ["work_dir"], "logp_diff"), (torch.nn.functional.log_softmax(-energy_pred/self.hyperparams.EKLloss_temperature, dim=0)-torch.nn.functional.log_softmax(-energy/self.hyperparams.EKLloss_temperature, dim=0)).detach().cpu().numpy())
                np.save(os.path.join(os.environ["work_dir"], "p_data"), (torch.nn.functional.softmax(-energy/self.hyperparams.EKLloss_temperature, dim=0)).detach().cpu().numpy())
                raise RuntimeError
        if self.hyperparams.mode is not None and "Energy-EMD" in self.hyperparams.mode:
            raise Exception("loss mode Energy-EMD is not implemented")
            energy_pred = self.ising_model.forward_soft(logits.reshape(B,H*W,self.model.alphabet_size))
            assert not torch.isinf(energy_pred).any()
            p_pred = torch.nn.functional.softmax(-energy_pred+energy_pred.min())
            p_seq = torch.nn.functional.softmax(-energy+energy.min())

            Eloss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(-energy_pred/self.hyperparams.Eloss_temperature, dim=0), torch.nn.functional.log_softmax(-energy/self.hyperparams.Eloss_temperature, dim=0), log_target=True, reduction="none")*B
            # Eloss = (energy_pred - energy)**2
            self.lg("energyloss", Eloss)
            losses += Eloss.mean()*self.hyperparams.prefactor_E
        if self.hyperparams.mode is not None and "FED" in self.hyperparams.mode:
            raise Exception("Training by FED loss doesn't seem to help")
            diff_at = H*W/2
            seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=self.model.alphabet_size).reshape(*shape, self.model.alphabet_size)
            self.RCL.buffer2rc_trajs(seq_onehot)
            m_seq = self.RCL.rc_trajs

            norm_logits = torch.nn.functional.softmax(logits, dim=-1)
            self.RCL.buffer2rc_trajs(norm_logits)
            m_pred = self.RCL.rc_trajs
            def soft_prob_greater(X, T, tau=1.0):
                # Use a sigmoid approximation to smooth the indicator function
                return torch.sigmoid((X - T) / tau)
            FED_pred = -torch.log(soft_prob_greater(torch.abs(m_pred), diff_at).mean()+1e-8)+torch.log(soft_prob_greater(diff_at, torch.abs(m_pred), ).mean()+1e-8)
            FED_seq =  -torch.log(soft_prob_greater(torch.abs(m_seq), diff_at).mean()+1e-8)+torch.log(soft_prob_greater(diff_at, torch.abs(m_seq), ).mean()+1e-8)
            FEDloss = torch.abs(FED_pred - FED_seq)
            self.lg("FEDloss", FEDloss.reshape(1))
            losses += FEDloss*self.hyperparams.prefactor_FED
        if self.hyperparams.mode is not None and ("RC" in self.hyperparams.mode or "RC-focal" in self.hyperparams.mode):
            ### Trainning by RC effectively trains a class-conditional model.
            xgrid = torch.linspace(-torch.prod(self.hyperparams.seq_dim), torch.prod(self.hyperparams.seq_dim), torch.prod(self.hyperparams.seq_dim)+1, device=logits.device)
            seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=self.model.alphabet_size).reshape(B,H,W,D,self.model.alphabet_size)
            self.RCL_seq.buffer2rc_trajs(seq_onehot)
            rc_seq = self.RCL_seq.rc_trajs
            kde_rc_seq = self.RCL_seq.kde(xgrid, 1., dump_hist=True)

            if self.loaded_uncond_model:
                self.RCL_logits.buffer2rc_trajs(norm_logits_sum.reshape(B,H,W,D,self.model.alphabet_size))
            else:
                self.RCL_logits.buffer2rc_trajs(norm_logits.reshape(B,H,W,D,self.model.alphabet_size))
            rc_logits = self.RCL_logits.rc_trajs
            kde_rc_logits= self.RCL_logits.kde(xgrid, 1., dump_hist=False)
            if "RC-focal" in self.hyperparams.mode:
                rc_loss = (1-kde_rc_logits)**2*torch.nn.functional.cross_entropy(kde_rc_logits.reshape([1,-1]), kde_rc_seq.reshape([1,-1]), reduction='none')
            else:
                rc_loss = torch.nn.functional.cross_entropy(kde_rc_logits.reshape([1,-1]), kde_rc_seq.reshape([1,-1]), reduction='none')
            self.lg("RCLoss", rc_loss)
            # rc_loss.sum().backward()
            # print(logits.grad)
            mse_rcloss = torch.abs(rc_logits-rc_seq)
            self.lg("MSERCLoss", mse_rcloss)
            losses += rc_loss.mean()*self.hyperparams.prefactor_RC
        if self.stage == "train":
            current_lr = self.optimizers().param_groups[0]['lr']
            self.lg("LR", torch.tensor([current_lr]))

        losses = losses.mean(-1)
        self.lg("loss", losses)

        if self.stage == "val":
            if self.hyperparams.model == "CNN3D":
                logits_pred, _ = self.dirichlet_flow_inference_3d(seq, cls, energy_op)
            elif self.hyperparams.model == "CNN2D":
                logits_pred, _ = self.dirichlet_flow_inference_2d(seq, cls, energy_op)
            seq_pred = torch.argmax(torch.concat([logits_pred, logits_pred.flip([-1])], dim=0), dim=-1)
            # torch.save(seq_pred, os.path.join(os.environ["work_dir"], f"seq_val"), )
            np.save(os.path.join(os.environ["work_dir"], f"seq_val"), seq_pred.cpu().detach().numpy(), )
            np.save(os.path.join(os.environ["work_dir"], f"logits_val"), torch.concat([logits_pred, logits_pred.flip([-1])], dim=0).cpu().detach().numpy())
        return losses.mean()

    def general_step(self, batch, batch_idx=None):
        seq, cls, energy, energy_op = batch
        ### Data augmentation by flipping the binary choices
        # if self.stage == "train":
        #     seq_symm = -seq+1
        #     seq = torch.cat([seq, seq_symm])
        #     probs = torch.cat([probs, probs])
            
        if self.hyperparams.model == "CNN3D":
            B, H, W, D = seq.shape
            xt, t = sample_cond_prob_path(self.hyperparams, seq, self.model.alphabet_size)
        elif self.hyperparams.model == "CNN2D":
            B, H, W = seq.shape
            xt, t = sample_cond_prob_path_2d(self.hyperparams, seq, self.model.alphabet_size)
        shape = seq.shape
        if self.hyperparams.cls_free_guidance:
            if self.hyperparams.guidance_op == "energy-magnetization":
                energy_op_inp = energy_op
                logits = self.model(xt, t, cls=cls, e=energy_op_inp)
                logits_symm = self.model(1-xt, t, cls=cls.max()-cls, e=energy_op_inp)
            elif self.hyperparams.guidance_op == "magnetization":
                # cls_inp = torch.where(torch.rand(B, device=self.device) >= self.hyperparams.cls_free_noclass_ratio, cls.squeeze(), 73) # set fraction of the classes to the unconditional class
                logits = self.model(xt, t, cls=cls)
                # cls_inp_symm = torch.where(torch.rand(B, device=self.device) >= self.hyperparams.cls_free_noclass_ratio, H*W*2-cls.squeeze(), 73) # set fraction of the classes to the unconditional class
                logits_symm = self.model(1-xt, t, cls=cls.max()-cls)
            else:
                raise Exception("Unrecognized guidance order parameter")
        else:
            logits = self.model(xt, t)
            logits_symm = self.model(1-xt, t)
        if self.loaded_uncond_model:
            uncond_logits = self.uncond_model(xt,t)
            uncond_logits = uncond_logits.permute(0,2,3,1).reshape(-1,self.model.alphabet_size)
        if self.hyperparams.model == "CNN3D":
            raise Exception("3D not implemented")
        elif self.hyperparams.model == "CNN2D":
            logits = (logits.permute(0,2,3,1)).reshape(-1, self.model.alphabet_size)
            logits_symm = (logits_symm.permute(0,2,3,1)).reshape(-1, self.model.alphabet_size)
        # logits.retain_grad()
        # CELoss = torch.nn.functional.cross_entropy(logits, seq.reshape(-1), reduction='none').reshape(B,-1)
        CELoss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(torch.flip(logits_symm, [-1]))*0.5
                                              +torch.nn.functional.log_softmax(logits)*0.5, seq.reshape(-1), reduction="none").reshape(B,-1)
        self.lg("CELoss", CELoss)
        if torch.isinf(torch.nn.functional.log_softmax(torch.flip(logits_symm, [-1]))).any():
            raise Exception("ERROR:: INF in: torch.log1p(-torch.exp(torch.nn.functional.log_softmax(logits_symm-logits_symm.max())))")
        if torch.isinf(torch.nn.functional.softmax(logits)).any():
            raise Exception("ERROR:: INF in: torch.nn.functional.softmax(logits)")
        losses_sym = self.hyperparams.prefactor_symm* torch.nn.functional.kl_div(torch.nn.functional.log_softmax(torch.flip(logits_symm, [-1])), 
                                                                        torch.nn.functional.log_softmax(logits), log_target=True, reduction="none").sum(-1).reshape(B,-1)
        self.lg("symloss", losses_sym)
        losses = self.hyperparams.prefactor_CE* CELoss + losses_sym
        
        norm_logits = torch.nn.functional.softmax(logits)
        if self.loaded_uncond_model:
            norm_uncond_logits = torch.nn.functional.softmax(uncond_logits)
            logits_sum = norm_logits*norm_uncond_logits
            norm_logits_sum = torch.nn.functional.softmax(logits_sum)
        if self.hyperparams.mode is not None and "Energy" in self.hyperparams.mode:
            if not self.loaded_uncond_model:
                energy_pred = self.ising_model.forward_soft(norm_logits.reshape(B,H*W,self.model.alphabet_size))
            else:
                energy_pred = self.ising_model.forward_hard(norm_logits_sum.reshape(B,H*W,self.model.alphabet_size))
            assert not torch.isinf(energy_pred).any()
            EKLloss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(-energy_pred/self.hyperparams.EKLloss_temperature, dim=0), torch.nn.functional.log_softmax(-energy/self.hyperparams.EKLloss_temperature, dim=0), log_target=True, reduction="none")*B
            MSEloss = torch.abs(energy_pred-energy)
            # Eloss = (energy_pred - energy)**2
            self.lg("energy_klloss", EKLloss)
            self.lg("eneergy_mseloss", MSEloss)
            losses += MSEloss.mean()*self.hyperparams.prefactor_eloss_mse + EKLloss.mean()*self.hyperparams.prefactor_EKL
            ### DEBUG
            if self.stage == "val" and B <= 1024:
                # np.save(os.path.join(os.environ["work_dir"], "EKLloss"), EKLloss.detach().cpu().numpy())
                np.save(os.path.join(os.environ["work_dir"], "energy"), energy.detach().cpu().numpy())
                np.save(os.path.join(os.environ["work_dir"], "energy_pred"), energy_pred.detach().cpu().numpy())
                np.save(os.path.join(os.environ["work_dir"], "logp_diff"), (torch.nn.functional.log_softmax(-energy_pred/self.hyperparams.EKLloss_temperature, dim=0)-torch.nn.functional.log_softmax(-energy/self.hyperparams.EKLloss_temperature, dim=0)).detach().cpu().numpy())
                np.save(os.path.join(os.environ["work_dir"], "p_data"), (torch.nn.functional.softmax(-energy/self.hyperparams.EKLloss_temperature, dim=0)).detach().cpu().numpy())
                raise RuntimeError
        if self.hyperparams.mode is not None and "Energy-EMD" in self.hyperparams.mode:
            raise Exception("loss mode Energy-EMD is not implemented")
            energy_pred = self.ising_model.forward_soft(logits.reshape(B,H*W,self.model.alphabet_size))
            assert not torch.isinf(energy_pred).any()
            p_pred = torch.nn.functional.softmax(-energy_pred+energy_pred.min())
            p_seq = torch.nn.functional.softmax(-energy+energy.min())

            Eloss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(-energy_pred/self.hyperparams.Eloss_temperature, dim=0), torch.nn.functional.log_softmax(-energy/self.hyperparams.Eloss_temperature, dim=0), log_target=True, reduction="none")*B
            # Eloss = (energy_pred - energy)**2
            self.lg("energyloss", Eloss)
            losses += Eloss.mean()*self.hyperparams.prefactor_E
        if self.hyperparams.mode is not None and "FED" in self.hyperparams.mode:
            raise Exception("Training by FED loss doesn't seem to help")
            diff_at = H*W/2
            seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=self.model.alphabet_size).reshape(*shape, self.model.alphabet_size)
            self.RCL.buffer2rc_trajs(seq_onehot)
            m_seq = self.RCL.rc_trajs

            norm_logits = torch.nn.functional.softmax(logits, dim=-1)
            self.RCL.buffer2rc_trajs(norm_logits)
            m_pred = self.RCL.rc_trajs
            def soft_prob_greater(X, T, tau=1.0):
                # Use a sigmoid approximation to smooth the indicator function
                return torch.sigmoid((X - T) / tau)
            FED_pred = -torch.log(soft_prob_greater(torch.abs(m_pred), diff_at).mean()+1e-8)+torch.log(soft_prob_greater(diff_at, torch.abs(m_pred), ).mean()+1e-8)
            FED_seq =  -torch.log(soft_prob_greater(torch.abs(m_seq), diff_at).mean()+1e-8)+torch.log(soft_prob_greater(diff_at, torch.abs(m_seq), ).mean()+1e-8)
            FEDloss = torch.abs(FED_pred - FED_seq)
            self.lg("FEDloss", FEDloss.reshape(1))
            losses += FEDloss*self.hyperparams.prefactor_FED
        if self.hyperparams.mode is not None and ("RC" in self.hyperparams.mode or "RC-focal" in self.hyperparams.mode):
            ### Trainning by RC effectively trains a class-conditional model.
            xgrid = torch.linspace(-torch.prod(self.hyperparams.seq_dim), torch.prod(self.hyperparams.seq_dim), torch.prod(self.hyperparams.seq_dim)+1, device=logits.device)
            seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=self.model.alphabet_size).reshape(B,H,W,self.model.alphabet_size)
            self.RCL_seq.buffer2rc_trajs(seq_onehot)
            rc_seq = self.RCL_seq.rc_trajs
            kde_rc_seq = self.RCL_seq.kde(xgrid, 1., dump_hist=True)

            if self.loaded_uncond_model:
                self.RCL_logits.buffer2rc_trajs(norm_logits_sum.reshape(B,H,W,self.model.alphabet_size))
            else:
                self.RCL_logits.buffer2rc_trajs(norm_logits.reshape(B,H,W,self.model.alphabet_size))
            rc_logits = self.RCL_logits.rc_trajs
            kde_rc_logits= self.RCL_logits.kde(xgrid, 1., dump_hist=False)
            if "RC-focal" in self.hyperparams.mode:
                rc_loss = (1-kde_rc_logits)**2*torch.nn.functional.cross_entropy(kde_rc_logits.reshape([1,-1]), kde_rc_seq.reshape([1,-1]), reduction='none')
            else:
                rc_loss = torch.nn.functional.cross_entropy(kde_rc_logits.reshape([1,-1]), kde_rc_seq.reshape([1,-1]), reduction='none')
            self.lg("RCLoss", rc_loss)
            # rc_loss.sum().backward()
            # print(logits.grad)
            mse_rcloss = torch.abs(rc_logits-rc_seq)
            self.lg("MSERCLoss", mse_rcloss)
            losses += rc_loss.mean()*self.hyperparams.prefactor_RC
        if self.stage == "train":
            current_lr = self.optimizers().param_groups[0]['lr']
            self.lg("LR", torch.tensor([current_lr]))

        losses = losses.mean(-1)
        self.lg("loss", losses)

        if self.stage == "val":
            if self.hyperparams.model == "CNN3D":
                raise Exception("ERROR:: 3D inference is not implemented")
            elif self.hyperparams.model == "CNN2D":
                if self.hyperparams.cls_free_guidance:
                    if self.hyperparams.guidance_op == "energy-magnetization":
                        logits_pred, _ = self.dirichlet_flow_inference_2d(seq, cls, energy_op)
                    elif self.hyperparams.guidance_op == "magnetization":
                        logits_pred, _ = self.dirichlet_flow_inference_2d(seq, cls, energy_op)
                    else:
                        raise Exception("Unrecognized guidance_op")
                else:
                    logits_pred, _ = self.dirichlet_flow_inference_2d(seq, cls, energy_op)
            seq_pred = torch.argmax(torch.concat([logits_pred, logits_pred.flip([-1])], dim=0), dim=-1)
            # torch.save(seq_pred, os.path.join(os.environ["work_dir"], f"seq_val"), )
            np.save(os.path.join(os.environ["work_dir"], f"seq_val"), seq_pred.cpu().detach().numpy(), )
            np.save(os.path.join(os.environ["work_dir"], f"logits_val"), torch.concat([logits_pred, logits_pred.flip([-1])], dim=0).cpu().detach().numpy())
        return losses.mean()
    
    def on_train_epoch_start(self):
        if self.hyperparams.cls_free_guidance and self.hyperparams.uncond_model_ckpt is not None and not self.loaded_uncond_model:
            hyperparams_uncond = deepcopy(self.hyperparams)
            hyperparams_uncond.cls_free_guidance = False
            self.uncond_model = CNNModel2D(hyperparams_uncond, self.hyperparams.channels, 2, 2)
            self.uncond_model.load_state_dict(upgrade_state_dict(
                torch.load(self.hyperparams.uncond_model_ckpt, map_location=self.device)['state_dict'],
                prefixes=['model.']))
            self.uncond_model.eval()
            self.uncond_model.to(self.device)
            self.loaded_uncond_model = True

    def training_step(self, batch, batch_idx):
        self.stage = "train"
        opt = self.optimizers()
        if self.hyperparams.lr_decay:
            # manually adjust lr
            adjust_learning_rate(opt, self.current_epoch, self.hyperparams.last_epoch, self.hyperparams.lr, 0.99)
        else:
            # manually delete scheduler in the Trainer
            for param_group in opt.param_groups:
                param_group['lr'] = self.hyperparams.lr  # Set to the new learning rate
        
        if self.hyperparams.model == "CNN2D":
            loss = self.general_step(batch)
        elif self.hyperparams.model == "CNN3D":
            loss = self.general_step_3d(batch)
        else:
            print("Skipping general step")
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def on_validation_epoch_start(self):
        if self.hyperparams.cls_free_guidance and self.hyperparams.uncond_model_ckpt is not None and not self.loaded_uncond_model:
            hyperparams_uncond = deepcopy(self.hyperparams)
            hyperparams_uncond.cls_free_guidance = False
            self.uncond_model = CNNModel2D(hyperparams_uncond, self.hyperparams.channels, 2, 2)
            self.uncond_model.load_state_dict(upgrade_state_dict(
                torch.load(self.hyperparams.uncond_model_ckpt, map_location=self.device)['state_dict'],
                prefixes=['model.']))
            self.uncond_model.eval()
            self.uncond_model.to(torch.float32).to(self.device)
            self.loaded_uncond_model = True

    def validation_step(self, batch, batch_idx):
        self.stage = "val"
        if self.hyperparams.model == "CNN2D":
            loss = self.general_step(batch, batch_idx)
        elif self.hyperparams.model == "CNN3D":
            loss = self.general_step_3d(batch, batch_idx)
        else:
            print("Skipping general step")


    def get_cls_guided_flow(self, xt, alpha, p_x0_given_xt):
        B, H, W, K = xt.shape
        # get the matrix of scores of the conditional probability flows for each simplex corner
        cond_scores_mats = ((alpha - 1) * (torch.eye(self.model.alphabet_size).to(xt)[None, :] / xt[..., None]))  # [B, H, W, K, K]
        cond_scores_mats = cond_scores_mats - cond_scores_mats.mean(3)[:, :, :, None, :]  # [B, H, W, K, K] now the columns sum up to 0
        assert torch.allclose(cond_scores_mats.sum(3), torch.zeros((B, H, W, K)).to(xt.device),atol=1e-4), cond_scores_mats.sum(3)
        Q_mats = cond_scores_mats.clone()  # [B, H, W, K, K]
        Q_mats[:, :, :, -1, :] = torch.ones((B, H, W, K))  # [B, H, W, K, K]

        score = torch.einsum('ijklm,ijkm->ijkl', cond_scores_mats, p_x0_given_xt)  # [B, H, W, K] add up the columns of conditional flow scores weighted by the predicted probability of each corner
        assert torch.allclose(score.sum(3), torch.zeros((B, H, W)).to(xt.device),atol=1e-4)
        ### for debugging whether these probabilities also have negative entries and are off of the simplex in other ways
        score_ = score.clone()  # [B, H, W, K]
        score_[:, :, :, -1] = torch.ones(B, H, W)  # [B, H, W, K]
        p_x0_given_xt_back = torch.linalg.solve(Q_mats, score_)
        if torch.isnan(p_x0_given_xt_back).any():
            raise Exception("NAN")
        if (p_x0_given_xt_back < 0).any():
            raise Exception("Negative probability")

        cls_score = self.get_cls_score(xt, alpha[None].expand(B))
        if self.hyperparams.scale_cls_score:
            norm_score = torch.norm(score, dim=3, keepdim=True)
            norm_cls_score = torch.norm(cls_score, dim=3, keepdim=True)
            cls_score = torch.where(norm_cls_score != 0, cls_score * norm_score / norm_cls_score, cls_score)
        guided_score = cls_score + score
        """
        ### test1: v=UD^{-1}(s+s')
        guided_score_ = guided_score.clone()  # [B, H, W, K]
        guided_score_[:, :, :, -1] = torch.ones(B, H, W)  # [B, H, W, K]
        p_x0_given_xt_y = torch.linalg.solve(Q_mats, guided_score_) # [B, H, W, K]
        """
        ### test2: v=UP+UD^{-1}s'
        
        cls_score_ = cls_score.clone()  # [B, H, W, K]
        cls_score_[:, :, :, -1] = torch.ones(B, H, W)  # [B, H, W, K]
        p_xt_given_y = torch.linalg.solve(Q_mats, cls_score_)
        p_x0_given_xt_y = p_xt_given_y + p_x0_given_xt



        if torch.isnan(p_x0_given_xt_y).any():
            print("WARNNING(NAN): there were this many nans in the probs_cond of the classifier score: ", torch.isnan(p_x0_given_xt_y).sum(), "We are setting them to 0.")
            p_x0_given_xt_y = torch.nan_to_num(p_x0_given_xt_y)
        return p_x0_given_xt_y
    
    def get_cls_score(self, xt, alpha):
        with torch.enable_grad():
            xt_ = xt.clone().detach().requires_grad_(True)
            xt_.requires_grad = True
            if self.hyperparams.cls_expanded_simplex:
                xt_, prior_weights = expand_simplex(xt, alpha, self.hyperparams.prior_pseudocount)
            if self.hyperparams.analytic_cls_score:
                B, H, W, K = xt.shape
                xt_ = xt_.reshape(B,H*W,K)

                x0_given_y = self.toy_data.data_class1.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1) # B, L, K, K
                xt_expanded = xt_.unsqueeze(1).expand(-1, x0_given_y.shape[1], -1, -1) 
                selected_xt = torch.gather(xt_expanded, dim=3, index=x0_given_y).squeeze()
                p_xt_given_x0_y = selected_xt ** (alpha[:,None,None])
                p_xt_given_y = p_xt_given_x0_y.mean(1)              

                x0_all = torch.cat([self.toy_data.data_class1, self.toy_data.data_class2], dim=0).unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)
                xt_expanded = xt_.unsqueeze(1).expand(-1, x0_all.shape[1], -1, -1)
                selected_xt = torch.gather(xt_expanded, dim=3, index=x0_all).squeeze()
                p_xt_given_x0 = selected_xt ** (alpha[:,None,None])
                p_xt = p_xt_given_x0.mean(1)                

                p_y_given_xt = p_xt_given_y/p_xt
                p_y_given_xt = p_y_given_xt.prod(-1)
                log_prob = torch.log(p_y_given_xt).sum()
                assert not torch.isnan(log_prob).any()
                cls_score = torch.autograd.grad(log_prob,[xt_])[0].reshape(B,H,W,K)
                assert not torch.isnan(cls_score).any()
            else:
                raise Exception("ERROR:: Don't have a cls_model")
                cls_logits = self.cls_model(xt_, t=alpha)
                loss = torch.nn.functional.cross_entropy(cls_logits, torch.ones(len(xt), dtype=torch.long, device=xt.device) * self.hyperparams.target_class).mean()
                assert not torch.isnan(loss).any()
                cls_score = - torch.autograd.grad(loss,[xt_])[0]  # need the minus because cross entropy loss puts a minus in front of log probability.
                assert not torch.isnan(cls_score).any()
        cls_score = cls_score - cls_score.mean(-1)[:,:,:,None]
        return cls_score.detach()

    def get_cls_free_guided_flow(self, xt, alpha, logits, logits_cond):
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs_cond = torch.nn.functional.softmax(logits_cond, dim=-1)

        assert (xt>=0.).all()
        # cond_scores_mats = ((alpha - 1) * (torch.eye(self.model.alphabet_size).to(xt)[None, :] / (xt[..., None]+1e-6)))  # [B, H, W, K, K]
        # cond_scores_mats = cond_scores_mats - cond_scores_mats.mean(3)[:, :, :, None, :]  # [B, H, W, K, K] now the columns sum up to 0

        # score = torch.einsum('ijklm,ijkm->ijkl', cond_scores_mats, probs)  # [B, H, W, K] add up the columns of conditional flow scores weighted by the predicted probability of each corner
        # score_cond = torch.einsum('ijklm,ijkm->ijkl', cond_scores_mats, probs_cond)  # [B, H, W, K] add up the columns of conditional flow scores weighted by the predicted probability of each corner
        # score_guided = (1 - self.hyperparams.guidance_scale) * score + self.hyperparams.guidance_scale * score_cond
        # if torch.isnan(score_guided).any():
        #     print("NAN")
        # Q_mats = cond_scores_mats.clone()  # [B, H, W, K, K]
        # Q_mats[:, :, :, -1, :] = torch.ones((B, H, W, K))  # [B, H, W, K, K]
        # score_guided_ = score_guided.clone()  # [B, H, W, K]
        # score_guided_[:, :, :, -1] = torch.ones(B, H, W)  # [B, H, W, K]
        # if torch.isnan(cond_scores_mats).any():
        #     print("NAN")
        # flow_guided = torch.linalg.solve(Q_mats, score_guided_)  # [B, H, W, K]
        # if torch.isnan(flow_guided).any():
        #     print("NAN")
        # return flow_guided
        if self.hyperparams.probability_tilt:
            ### probability tilt
            if self.hyperparams.probability_tilt_scheduled:
                g = self.hyperparams.guidance_scale * (alpha-1)/(self.hyperparams.alpha_max-1)
                flow_guided = probs_cond ** g * probs
                flow_guided = flow_guided / flow_guided.sum(-1)[...,None]
            else:
                flow_guided = probs_cond ** self.hyperparams.guidance_scale * probs
                flow_guided = flow_guided / flow_guided.sum(-1)[...,None]
        else:
            ### probability addition
            score_guided_additional = ((1 - self.hyperparams.guidance_scale) * (probs - probs_cond) )
            flow_guided = probs_cond + score_guided_additional
        return flow_guided

    @torch.no_grad()
    def dirichlet_flow_inference_2d(self, seq, cls, energy_op):

        B, H, W = seq.shape
        K = self.model.alphabet_size
        x0 = torch.distributions.Dirichlet(torch.ones(B, H, W, K, device=seq.device)).sample()

        eye = torch.eye(K).to(x0)
        xt_out = []
        xt = x0.clone()
        # xt_out.append( xt.detach().cpu())
        np.save(os.path.join(os.environ["work_dir"], f"logits_val_inttime{1.00}"), torch.concat([xt, xt.flip([-1])], dim=0).cpu().to(torch.float16))
        # return xt, x0
        t_span = torch.linspace(1, self.hyperparams.alpha_max, self.hyperparams.num_integration_steps, device = self.device)
        for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            # xt_expanded, prior_weights = expand_simplex(xt, s[None].expand(B), self.hyperparams.prior_pseudocount)
            skip = False
            if torch.rand(1) > (1-self.hyperparams.shuffle_cls_freq):
                # skip=True
                # continue
                if not self.hyperparams.enforce_symm:
                    logits = self.model(xt, t=s[None].expand(B), cls=cls[torch.randperm(B)], e=energy_op[torch.randperm(B)])
                else:
                    ### TODO: Something wrong with the following symmetric prediction,
                    ## either the model can't do this because not trained with symm loss,
                    ## or symm op seperately on cond and uncond models doesn't work.
                    ## Can consider putting the enforce_symm op to after the score guidance.
                    raise Exception("Score guided generation doesn't work with enforce_symm == True")
                    logits = self.model(xt[:B//2], t=s[None].expand(B//2), cls=cls[:B//2][torch.randperm(B//2)], e=energy_op[:B//2][torch.randperm(B//2)])
                    logits = torch.cat([torch.flip(logits, [1]), logits], dim=0)
                # raise Exception("Shuffling")
            else:
                if not self.hyperparams.enforce_symm:
                    logits = self.model(xt[:B], t=s[None].expand(B), cls=cls[:B], e=energy_op[:B])
                else:
                    ### TODO: Something wrong with the following symmetric prediction,
                    ## either the model can't do this because not trained with symm loss,
                    ## or symm op seperately on cond and uncond models doesn't work.
                    ## Can consider putting the enforce_symm op to after the score guidance.
                    raise Exception("Score guided generation doesn't work with enforce_symm == True")
                    logits = self.model(xt[:B//2], t=s[None].expand(B//2), cls=cls[:B//2], e=energy_op[:B//2])
                    logits = torch.cat([torch.flip(logits, [1]), logits], dim=0)
            if self.hyperparams.score_free_guidance or not self.hyperparams.cls_free_guidance:
                flow_probs = torch.nn.functional.softmax(logits.permute(0,2,3,1)/self.hyperparams.flow_temp, -1)
            else:
                # probs_cond = torch.nn.functional.softmax(logits.permute(0,2,3,1)/self.hyperparams.flow_temp, -1)
                if self.hyperparams.uncond_model_ckpt is not None:
                    if not self.hyperparams.enforce_symm:
                        logits_uncond = self.uncond_model(xt.float(), t=s[None].expand(B).float())
                    else:
                        ### TODO: Something wrong with the following symmetric prediction,
                        ## either the model can't do this because not trained with symm loss,
                        ## or symm op seperately on cond and uncond models doesn't work.
                        ## Can consider putting the enforce_symm op to after the score guidance.
                        raise Exception("Score guided generation doesn't work with enforce_symm == True")
                        logits_uncond = self.uncond_model(xt[:B//2].float(), t=s[None].expand(B//2).float()).double()
                        logits_uncond = torch.cat([torch.flip(logits_uncond, [1]), logits_uncond], dim=0)
                else:
                    if not self.hyperparams.enforce_symm:
                        logits_uncond = self.model(xt, t=s[None].expand(B), cls=torch.ones(B, device=self.device).to(torch.int64)*(73))
                    else:
                        ### TODO: Something wrong with the following symmetric prediction,
                        ## either the model can't do this because not trained with symm loss,
                        ## or symm op seperately on cond and uncond models doesn't work.
                        ## Can consider putting the enforce_symm op to after the score guidance.
                        raise Exception("Score guided generation doesn't work with enforce_symm == True")
                        logits_uncond = self.model(xt[:B//2], t=s[None].expand(B//2), cls=torch.ones(B//2, device=self.device).to(torch.int64)*(73))
                        logits_uncond = torch.cat([torch.flip(logits_uncond, [1]), logits_uncond], dim=0)
                # probs_uncond = torch.nn.functional.softmax(logits_uncond.permute(0,2,3,1)/self.hyperparams.flow_temp, -1)
                if skip:
                    flow_probs = torch.nn.functional.softmax(logits_uncond.permute(0,2,3,1), dim=-1)
                else:
                    flow_probs = self.get_cls_free_guided_flow(xt, s+1e-4, logits_uncond.permute(0,2,3,1), logits.permute(0,2,3,1))
                

            if self.hyperparams.cls_guidance:
                probs_cond = self.get_cls_guided_flow(xt, s + 1e-4, flow_probs)
                flow_probs = probs_cond * self.hyperparams.guidance_scale + flow_probs * (1 - self.hyperparams.guidance_scale)

            default_dtype = flow_probs.dtype
            if not torch.allclose((flow_probs.reshape(B,-1,K)).sum(2), torch.ones((B, H*W), device=self.device, dtype=default_dtype), atol=1e-4) or not (flow_probs >= 0).all():
                print(f'WARNING: flow_probs.min(): {flow_probs.min()}. Some values of flow_probs do not lie on the simplex. There are we are {(flow_probs<0).sum()} negative values in flow_probs of shape {flow_probs.shape} that are negative. We are projecting them onto the simplex.')
                flow_probs = simplex_proj(flow_probs.reshape(B,-1)).reshape(B,H,W,K)
            assert not torch.isnan(flow_probs).any()
            c_factor = self.condflow.c_factor(xt.cpu().detach().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(xt).to(torch.float32)
            assert not torch.isnan(c_factor).any()
            assert not torch.isinf(c_factor).any()

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
            assert not torch.isnan(cond_flows).any()
            assert not torch.isinf(cond_flows).any()
            # V=U*P: flow = conditional_flow*probability_path
            flow = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1)
            # flow_uncond = (probs_uncond.unsqueeze(-2) * cond_flows).sum(-1)

            xt = xt + flow * (t - s) 
            assert not torch.isnan(xt).any()
            if not torch.allclose((xt.reshape(B,-1,K)).sum(2), torch.ones((B, H*W), device=self.device), atol=1e-4) or not (xt >= 0).all():
                print(f'WARNING@time{s}: xt.min(): {xt.min()}. Some values of xt do not lie on the simplex. There are we are {(xt<0).sum()} negative values in xt of shape {xt.shape} that are negative. We are projecting them onto the simplex.')
                xt = simplex_proj(xt.reshape(B,-1)).reshape(B,H,W,K)
            if (i+1) % self.hyperparams.dump_freq == 0:
                # xt_out.append( xt.detach().cpu())
                # torch.save(xt.detach(), os.path.join(os.environ["work_dir"], f"logits_val_inttime{t}"), )
                np.save(os.path.join(os.environ["work_dir"], f"logits_val_inttime{t}"), torch.concat([xt, xt.flip([-1])], dim=0).cpu().detach().numpy())
                # np.save(os.path.join(os.environ["work_dir"], f"flowprobs_val_inttime{t}"), flow_probs.cpu().detach().numpy())
        np.save(os.path.join(os.environ["work_dir"], f"logits_val_inttime{t_span[-1]}"), torch.concat([xt, xt.flip([-1])], dim=0).detach().cpu().numpy(), )
        # torch.save(xt_out,  os.path.join(os.environ["work_dir"], f"logits_val_all"))
               
        return xt, x0


    @torch.no_grad()
    def dirichlet_flow_inference_3d(self, seq, cls, energy_op):

        B, H, W, D = seq.shape
        K = self.model.alphabet_size
        x0 = torch.distributions.Dirichlet(torch.ones(B, H, W, D, K, device=seq.device)).sample()

        eye = torch.eye(K).to(x0)
        xt_out = []
        xt = x0.clone()
        # xt_out.append( xt.detach().cpu())
        np.save(os.path.join(os.environ["work_dir"], f"logits_val_inttime{1.00}"), torch.concat([xt, xt.flip([-1])], dim=0).cpu().to(torch.float16))
        # return xt, x0
        t_span = torch.linspace(1, self.hyperparams.alpha_max, self.hyperparams.num_integration_steps, device = self.device)
        for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            # xt_expanded, prior_weights = expand_simplex(xt, s[None].expand(B), self.hyperparams.prior_pseudocount)
            skip = False
            if torch.rand(1) > (1-self.hyperparams.shuffle_cls_freq):
                # skip=True
                # continue
                logits = self.model(xt, t=s[None].expand(B), cls=cls[torch.randperm(B)], e=energy_op[torch.randperm(B)])
            else:
                logits = self.model(xt[:B], t=s[None].expand(B), cls=cls[:B], e=energy_op[:B])

            if self.hyperparams.score_free_guidance or not self.hyperparams.cls_free_guidance:
                flow_probs = torch.nn.functional.softmax(logits.permute(0,2,3,4,1)/self.hyperparams.flow_temp, -1)
            else:
                # probs_cond = torch.nn.functional.softmax(logits.permute(0,2,3,4,1)/self.hyperparams.flow_temp, -1)
                if self.hyperparams.uncond_model_ckpt is not None:
                    logits_uncond = self.uncond_model(xt.float(), t=s[None].expand(B).float())

                else:
                    logits_uncond = self.model(xt, t=s[None].expand(B), cls=torch.ones(B, device=self.device).to(torch.int64)*(73))

                # probs_uncond = torch.nn.functional.softmax(logits_uncond.permute(0,2,3,4,1)/self.hyperparams.flow_temp, -1)
                if skip:
                    flow_probs = torch.nn.functional.softmax(logits_uncond.permute(0,2,3,4,1), dim=-1)
                else:
                    flow_probs = self.get_cls_free_guided_flow(xt, s+1e-4, logits_uncond.permute(0,2,3,4,1), logits.permute(0,2,3,4,1))
                

            if self.hyperparams.cls_guidance:
                probs_cond = self.get_cls_guided_flow(xt, s + 1e-4, flow_probs)
                flow_probs = probs_cond * self.hyperparams.guidance_scale + flow_probs * (1 - self.hyperparams.guidance_scale)

            default_dtype = flow_probs.dtype
            if not torch.allclose((flow_probs.reshape(B,-1,K)).sum(2), torch.ones((B, H*W*D), device=self.device, dtype=default_dtype), atol=1e-4) or not (flow_probs >= 0).all():
                print(f'WARNING: flow_probs.min(): {flow_probs.min()}. Some values of flow_probs do not lie on the simplex. There are we are {(flow_probs<0).sum()} negative values in flow_probs of shape {flow_probs.shape} that are negative. We are projecting them onto the simplex.')
                flow_probs = simplex_proj(flow_probs.reshape(B,-1)).reshape(B,H,W,D,K)
            assert not torch.isnan(flow_probs).any()
            c_factor = self.condflow.c_factor(xt.cpu().detach().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(xt).to(torch.float32)
            assert not torch.isnan(c_factor).any()
            assert not torch.isinf(c_factor).any()

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
            assert not torch.isnan(cond_flows).any()
            assert not torch.isinf(cond_flows).any()
            # V=U*P: flow = conditional_flow*probability_path
            flow = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1)
            # flow_uncond = (probs_uncond.unsqueeze(-2) * cond_flows).sum(-1)

            xt = xt + flow * (t - s) 
            assert not torch.isnan(xt).any()
            if not torch.allclose((xt.reshape(B,-1,K)).sum(2), torch.ones((B, H*W*D), device=self.device), atol=1e-4) or not (xt >= 0).all():
                print(f'WARNING@time{s}: xt.min(): {xt.min()}. Some values of xt do not lie on the simplex. There are we are {(xt<0).sum()} negative values in xt of shape {xt.shape} that are negative. We are projecting them onto the simplex.')
                xt = simplex_proj(xt.reshape(B,-1)).reshape(B,H,W,D,K)
            if (i+1) % self.hyperparams.dump_freq == 0:
                # xt_out.append( xt.detach().cpu())
                # torch.save(xt.detach(), os.path.join(os.environ["work_dir"], f"logits_val_inttime{t}"), )
                np.save(os.path.join(os.environ["work_dir"], f"logits_val_inttime{t}"), torch.concat([xt, xt.flip([-1])], dim=0).cpu().detach().numpy())
                # np.save(os.path.join(os.environ["work_dir"], f"flowprobs_val_inttime{t}"), flow_probs.cpu().detach().numpy())
        np.save(os.path.join(os.environ["work_dir"], f"logits_val_inttime{t_span[-1]}"), torch.concat([xt, xt.flip([-1])], dim=0).detach().cpu().numpy(), )
        # torch.save(xt_out,  os.path.join(os.environ["work_dir"], f"logits_val_all"))
               
        return xt, x0

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


    def plot_probability_path(self, t, xt):
        pass

    def lg(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        log = self._log
        log[self.stage + "_" + key].extend(data)
