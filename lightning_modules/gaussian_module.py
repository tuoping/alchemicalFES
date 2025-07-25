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

def sample_cond_vector_field_2d(hyperparams, seq, channels):
    shape = seq.shape
    batchsize = seq.shape[0]
    seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=channels).reshape(*shape, channels)
    sample_x = torch.randn(size=seq_onehot.shape, device=seq.device)

    
    t = 1 - (torch.from_numpy(scipy.stats.expon().rvs(size=batchsize)*hyperparams.time_scale).to(seq.device).float())
    sigma_t = 1-(1-hyperparams.sigma_min)*t
    sample_x *= sigma_t[:,None,None,None]
    sample_x += t[:,None,None,None]*seq_onehot

    ut = (seq_onehot - (1-hyperparams.sigma_min)*sample_x)/sigma_t[:,None,None,None]
    sample_x.requires_grad = False
    return sample_x, t, ut.float()


def RC(logits):
    assert logits.shape[-1] == 2
    B = logits.shape[0]
    RC = torch.sum(logits*torch.tensor([-1,1], device=logits.device)[None,None,None,:], dim=-1)
    RC = torch.sum(RC.reshape(B, -1), dim=-1)
    return RC.reshape(-1,1)


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
        seq, cls = batch
        ### Data augmentation by flipping the binary choices
        # if self.stage == "train":
        #     seq_symm = -seq+1
        #     seq = torch.cat([seq, seq_symm])
        #     cls = torch.cat([cls, cls])
        B = seq.shape[0]
        if self.hyperparams.model == "CNN3D":
            xt, t, ut = sample_cond_vector_field(self.hyperparams, seq, self.model.alphabet_size)
        elif self.hyperparams.model == "CNN2D":
            xt, t, ut = sample_cond_vector_field_2d(self.hyperparams, seq, self.model.alphabet_size)

        
        ut_model = self.model(xt, t, cls=cls)
        ut_model_symm = self.model(-xt, t, cls=cls)
        if self.hyperparams.model == "CNN3D":
            ut_model = ut_model.permute(0,2,3,4,1)
            ut_model_symm = ut_model_symm.permute(0,2,3,4,1)         
        elif self.hyperparams.model == "CNN2D":
            ut_model = ut_model.permute(0,2,3,1)
            ut_model_symm = ut_model_symm.permute(0,2,3,1)

        if self.hyperparams.model == "CNN3D":
            raise Exception("3D for symmetric model not implemented")
            losses = torch.norm((ut_model).reshape(B, -1, self.model.alphabet_size) - ut.reshape(B, -1, self.model.alphabet_size), dim=-1)**2/2.
        elif self.hyperparams.model == "CNN2D":
            if self.hyperparams.mode is not None and "focal" in self.hyperparams.mode:
                losses = torch.norm(((ut_model*0.5-ut_model_symm*0.5)).reshape(B, -1, self.model.alphabet_size) - ut.reshape(B, -1, self.model.alphabet_size), dim=-1)**2/2.*torch.norm(((ut_model*0.5-ut_model_symm*0.5)).reshape(B, -1, self.model.alphabet_size) - ut.reshape(B, -1, self.model.alphabet_size), dim=-1)**2
            else:
                losses = torch.norm(((ut_model*0.5-ut_model_symm*0.5)).reshape(B, -1, self.model.alphabet_size) - ut.reshape(B, -1, self.model.alphabet_size), dim=-1)**2/2.

        losses = losses.mean(-1)
        losses_sym = self.hyperparams.alpha*((ut_model+ut_model_symm)**2/2.).reshape(B,-1).mean(-1)
        self.lg("loss", losses)
        self.lg("symloss", losses_sym)
        losses = losses*self.hyperparams.prefactor_CE + losses_sym

        if self.hyperparams.mode is not None and "RC" in self.hyperparams.mode:
            u_rc_model = RC(ut_model*0.5-ut_model_symm*0.5)
            u_rc = RC(ut)
            rc_loss = ((u_rc_model-u_rc)**2/2*(u_rc_model-u_rc)**2).reshape(-1)
            self.lg("rcloss", rc_loss)
            losses += rc_loss.mean()*self.hyperparams.prefactor_RC

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
        xx = torch.normal(0, 1*self.hyperparams.time0_scale, size=(B,H,W,K), device=self.device)
        np.save(os.path.join(os.environ["work_dir"], "logits_val_inttime%.2f"%0.0), xx.cpu())
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
            u_t_sym = self.model(-xx, samples_tt)
            xx = xx + ((u_t*0.5-u_t_sym*0.5)).permute(0,2,3,1)*1./self.hyperparams.num_integration_steps

            xx_t.append(xx)
            np.save(os.path.join(os.environ["work_dir"], "logits_val_inttime%.2f"%(tt)), xx.detach().cpu().numpy())
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
