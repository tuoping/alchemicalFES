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
    t = torch.from_numpy(scipy.stats.expon().rvs(size=batchsize)*hyperparams.time_scale).to(seq.device).float()

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

    def forward(self, x):
        B, seq_len, K = x.shape
        assert seq_len == self.L**2
        edge_index = self.edge_index.to(x.device)
        # Get the spin values for the edge pairs
        spin_values = torch.argmax(x, dim=-1) * 2 - 1  # Convert one-hot encoding to -1, 1
        total_energy = 0.
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

# Function to manually adjust the learning rate
def adjust_learning_rate(optimizer, epoch, last_epoch, initial_lr, decay_factor):
    new_lr = initial_lr * (decay_factor ** ((epoch - last_epoch)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

class simplexModule(GeneralModule):
    def __init__(self, channels, num_cls, hyperparams, toy_data):
        super().__init__(hyperparams)
        self.load_model(channels, num_cls, hyperparams)
        self.condflow = DirichletConditionalFlow(K=self.model.alphabet_size, alpha_spacing=0.001, alpha_max=hyperparams.alpha_max)
        self.hyperparams = hyperparams
        self.RCL = RCLoss(RC)
        self.ising_model = IsingGNN(hyperparams.seq_dim[0])

    def load_model(self, channels, num_cls, hyperparams):
        if hyperparams.model == "CNN3D":
            self.model = CNNModel3D(hyperparams, channels, num_cls, classifier=True)
        elif hyperparams.model == "CNN2D":
            self.model = CNNModel2D(hyperparams, channels, num_cls, classifier=True)
        else:
            raise Exception("Unrecognized model type")

    def validation_step(self, batch, batch_idx):
        self.stage = 'val'
        loss = self.general_step(batch, batch_idx)
        self.log('val_loss', torch.tensor(self._log["val_loss"]).mean(), prog_bar=True)


    def general_step(self, batch, batch_idx=None):
        seq, probs = batch
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
        # self.plot_probability_path(t, xt)
        logits = self.model(xt, t, cls=None)
        logits_symm = self.model(1-xt, t, cls=None)
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
        losses_sym = self.hyperparams.alpha* torch.nn.functional.kl_div(torch.nn.functional.log_softmax(torch.flip(logits_symm, [-1])), 
                                                                        torch.nn.functional.log_softmax(logits), log_target=True, reduction="none").sum(-1).reshape(B,-1)
        self.lg("symloss", losses_sym)
        losses = self.hyperparams.prefactor_CE* CELoss + losses_sym
        if self.hyperparams.mode is not None and "RC" in self.hyperparams.mode:
            xgrid = torch.linspace(-36, 36, 36+1, device=logits.device)
            seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=self.model.alphabet_size).reshape(*shape, self.model.alphabet_size)
            self.RCL.buffer2rc_trajs(seq_onehot)
            rc_seq = self.RCL.kde(xgrid, 1., dump_hist=True)

            norm_logits = torch.nn.functional.softmax(logits, dim=-1)
            self.RCL.buffer2rc_trajs(norm_logits)
            if self.stage == "val":
                rc_logits= self.RCL.kde(xgrid, 1., dump_hist=False)
                if "focal" in self.hyperparams.mode:
                    rc_loss = (1-rc_logits)**2*(-rc_seq*torch.log(rc_logits+1e-12)).reshape(1,-1)
                else:
                    rc_loss = (-rc_seq*torch.log(rc_logits+1e-12)).reshape(1,-1)
                np.save(os.path.join(os.environ["work_dir"], f"logits_train"), norm_logits.cpu())
            else:
                rc_logits= self.RCL.kde(xgrid, 1.)
                if "focal" in self.hyperparams.mode:
                    # rc_loss = (1-rc_logits)**2*(-rc_seq*torch.log(rc_logits+1e-12)).reshape(1,-1)
                    rc_loss = (1-rc_logits)**2*torch.nn.functional.cross_entropy(rc_logits.reshape([1,-1]), rc_seq.reshape([1,-1]), reduction='none')
                else:
                    # rc_loss = rc_loss = (-rc_seq*torch.log(rc_logits+1e-12)).reshape(1,-1)
                    rc_loss = torch.nn.functional.cross_entropy(rc_logits.reshape([1,-1]), rc_seq.reshape([1,-1]), reduction='none')
            self.lg("RCLoss", rc_loss)
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
                raise Exception("ERROR:: 3D inference is not implemented")
            elif self.hyperparams.model == "CNN2D":
                logits_pred, _ = self.dirichlet_flow_inference_2d(seq)
            seq_pred = torch.argmax(logits_pred, dim=-1)
            np.save(os.path.join(os.environ["work_dir"], f"seq_val"), seq_pred.cpu())
            np.save(os.path.join(os.environ["work_dir"], f"logits_val"), logits_pred.cpu())
        return losses.mean()
    
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
        
        
        loss = self.general_step(batch)
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.stage = "val"
        loss = self.general_step(batch, batch_idx)


    def get_cls_guided_flow(self, xt, alpha, p_x0_given_xt):
        B, H, W, K = xt.shape
        # get the matrix of scores of the conditional probability flows for each simplex corner
        cond_scores_mats = ((alpha - 1) * (torch.eye(self.model.alphabet_size).to(xt)[None, :] / xt[..., None]))  # [B, H, W, K, K]
        cond_scores_mats = cond_scores_mats - cond_scores_mats.mean(3)[:, :, :, None, :]  # [B, H, W, K, K] now the columns sum up to 0
        # assert torch.allclose(cond_scores_mats.sum(2), torch.zeros((B, H, W, K)),atol=1e-4), cond_scores_mats.sum(2)

        score = torch.einsum('ijklm,ijkm->ijkl', cond_scores_mats, p_x0_given_xt)  # [B, H, W, K] add up the columns of conditional flow scores weighted by the predicted probability of each corner
        # assert torch.allclose(score.sum(2), torch.zeros((B, H, W)),atol=1e-4)

        cls_score = self.get_cls_score(xt, alpha[None].expand(B))
        if self.hyperparams.scale_cls_score:
            norm_score = torch.norm(score, dim=3, keepdim=True)
            norm_cls_score = torch.norm(cls_score, dim=3, keepdim=True)
            cls_score = torch.where(norm_cls_score != 0, cls_score * norm_score / norm_cls_score, cls_score)
        guided_score = self.hyperparams.grad_scale * cls_score + score * (1-self.hyperparams.grad_scale)

        Q_mats = cond_scores_mats.clone()  # [B, H, W, K, K]
        Q_mats[:, :, :, -1, :] = torch.ones((B, H, W, K))  # [B, H, W, K, K]
        guided_score_ = guided_score.clone()  # [B, H, W, K]
        guided_score_[:, :, :, -1] = torch.ones(B, H, W)  # [B, H, W, K]
        # p_x0_given_xt_y = torch.linalg.solve(Q_mats, guided_score_) # [B, H, W, K]
        p_x0_given_xt_y = torch.linalg.solve(Q_mats, guided_score) # [B, H, W, K]
        """
        # for debugging whether these probabilities also have negative entries and are off of the simplex in other ways
        cls_score_ = cls_score.clone()  # [B, L, K]
        cls_score_[:, :, -1] = torch.ones(B, L)  # [B, L, K]
        p_xt_given_y = torch.linalg.solve(Q_mats, cls_score_)

        score_guided_ = score.clone()  # [B, L, K]
        score_guided_[:, :, -1] = torch.ones(B, L)  # [B, L, K]
        p_x0_given_xt_back = torch.linalg.solve(Q_mats, score_guided_)
        """
        if torch.isnan(p_x0_given_xt_y).any():
            print("WARNNING(NAN): there were this many nans in the probs_cond of the classifier score: ", torch.isnan(p_x0_given_xt_y).sum(), "We are setting them to 0.")
            p_x0_given_xt_y = torch.nan_to_num(p_x0_given_xt_y)
        return p_x0_given_xt_y, cls_score
    
    def get_cls_score(self, xt, alpha):
        with torch.enable_grad():
            xt_ = xt.clone().detach().requires_grad_(True)
            xt_.requires_grad = True
            if self.hyperparams.cls_expanded_simplex:
                xt_, prior_weights = expand_simplex(xt, alpha[None].expand(xt_.shape[0]), self.args.prior_pseudocount)
            if self.hyperparams.analytic_cls_score:
                B, H, W, K = xt.shape
                log_prob = -self.ising_model(xt_.reshape(B,H*W,K)).sum()
                cls_score = torch.autograd.grad(log_prob, [xt_])[0]*(alpha[:, None, None, None] - 1)
            else:
                raise Exception("ERROR:: Don't have a cls_model")
                cls_logits = self.cls_model(xt_, t=alpha)
                loss = torch.nn.functional.cross_entropy(cls_logits, torch.ones(len(xt), dtype=torch.long, device=xt.device) * self.hyperparams.target_class).mean()
                assert not torch.isnan(loss).any()
                cls_score = - torch.autograd.grad(loss,[xt_])[0]  # need the minus because cross entropy loss puts a minus in front of log probability.
                assert not torch.isnan(cls_score).any()
        cls_score = cls_score - cls_score.mean(-1)[:,:,:,None]
        return cls_score.detach()

    @torch.no_grad()
    def dirichlet_flow_inference_2d(self, seq):

        B, H, W = seq.shape
        K = self.model.alphabet_size
        x0 = torch.distributions.Dirichlet(torch.ones(B, H, W, K, device=seq.device)).sample()

        eye = torch.eye(K).to(x0)
        xt = x0.clone()
        np.save(os.path.join(os.environ["work_dir"], f"logits_val_inttime{1.00}"), xt.cpu().to(torch.float16))
        # return xt, x0
        t_span = torch.linspace(1, self.hyperparams.alpha_max, self.hyperparams.num_integration_steps, device = self.device)
        for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            logits = self.model(xt, t=s[None].expand(B))
            flow_probs = torch.nn.functional.softmax(logits.permute(0,2,3,1)/self.hyperparams.flow_temp, -1)
            if self.hyperparams.cls_guidance:
                probs_cond, cls_score = self.get_cls_guided_flow(xt, s + 1e-4, flow_probs)
                flow_probs = probs_cond * self.hyperparams.guidance_scale + flow_probs * (1 - self.hyperparams.guidance_scale)


            default_dtype = flow_probs.dtype
            if not torch.allclose((flow_probs.reshape(B,-1,K)).sum(2), torch.ones((B, H*W), device=self.device, dtype=default_dtype), atol=1e-4) or not (flow_probs >= 0).all():
                print(f'WARNING: flow_probs.min(): {flow_probs.min()}. Some values of flow_probs do not lie on the simplex. There are we are {(flow_probs<0).sum()} negative values in flow_probs of shape {flow_probs.shape} that are negative. We are projecting them onto the simplex.')
                flow_probs = simplex_proj(flow_probs.reshape(B,-1)).reshape(B,H,W,K)

            c_factor = self.condflow.c_factor(xt.cpu().detach().numpy(), s.item())
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
                print(f'WARNING@time{s}: xt.min(): {xt.min()}. Some values of xt do not lie on the simplex. There are we are {(xt<0).sum()} negative values in xt of shape {xt.shape} that are negative. We are projecting them onto the simplex.')
                xt = simplex_proj(xt.reshape(B,-1)).reshape(B,H,W,K)
            np.save(os.path.join(os.environ["work_dir"], f"logits_val_inttime{t}"), xt.cpu().detach().numpy())
               
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
