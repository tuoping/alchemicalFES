
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import torch
import os,sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

os.environ["MODEL_DIR"]=f"logs-dir-ising/latt6x6T3.2/kernel3x3_timeembed_symmetrized/clsfreeG/ElossMloss/guidanceEM/prefrc10"
os.environ["work_dir"]=os.environ["MODEL_DIR"]

dataset_dir = ["data/ising-latt6x6-T2.2/buffer.npy"]

stage = "train"

channels = 2
seq_len= 6*6
seq_dim = (6,6)

ckpt = None
ckpt_epoch = 250
import glob
print(os.path.join(os.environ["MODEL_DIR"], f"model-epoch={ckpt_epoch}-train_loss=*"))
ckpt = glob.glob(os.path.join(os.environ["MODEL_DIR"], f"model-epoch={ckpt_epoch}-train_loss=*"))[0]
if stage == "train":
    batch_size = 1024
    if ckpt is not None: 
        print("Starting from ckpt:: ", ckpt)
elif stage == "val":
    batch_size = 1000
    if ckpt is None: 
        raise Exception("ERROR:: ckpt not initiated")
    print("Validating with ckpt::", ckpt)
else:
    raise Exception("Unrecognized stage")
num_workers = 0
max_steps = 40000000
max_epochs = 400
limit_train_batches = None
if stage == "train":
    limit_val_batches = 0.0
else:
    limit_val_batches = 100
# grad_clip = 1.
wandb = False
check_val_every_n_epoch = None
val_check_interval = None


class dataset_params():
    def __init__(self, toy_seq_len, toy_seq_dim, toy_simplex_dim, dataset_dir):
        self.toy_seq_len = toy_seq_len
        self.toy_seq_dim = toy_seq_dim
        self.toy_simplex_dim = toy_simplex_dim
        self.dataset_dir = dataset_dir
        self.cls_ckpt = None
        self.subset_size = None
        self.scale_magn = 1
        
dparams = dataset_params(seq_len, seq_dim, channels, dataset_dir)

from utils.dataset import IsingDataset_mixT
dparams.dataset_dir = dataset_dir
train_ds = IsingDataset_mixT(dparams)
if isinstance(dataset_dir, list):
    dparams.dataset_dir = os.path.join(dataset_dir[0], "val")
else:
    dparams.dataset_dir = os.path.join(dataset_dir, "val")

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = None
# val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

class Hyperparams():
    def __init__(self, mode=None, hidden_dim=16, num_cnn_stacks=1, lr=5e-4, dropout=0.0, clean_data=False, model="MLP"):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.kernel_size = 3
        self.padding = 1

        self.guided = True
        self.cls_free_guidance = True
        self.guidance_op = "energy-magnetization"
        self.uncond_model_ckpt = None
        self.cls_free_noclass_ratio = 0.0

        self.clean_data = clean_data
        self.num_cnn_stacks = num_cnn_stacks
        self.lr_decay=True
        self.lr = lr
        self.last_epoch = 0
        self.wandb = False
        self.seq_dim = torch.tensor(seq_dim).to("cuda")
        self.channels = channels
        self.model = model
        self.mode = mode
        self.gamma_focal = 2.
        self.prefactor_CE = 1.
        if mode is not None and ("RC" in mode or "RC-focal" in mode):
            self.prefactor_RC = 5
            self.prefactor_CE = 1
        if mode is not None and "Energy" in mode:
            self.prefactor_EKL = 1.
            self.EKLloss_temperature = 500
            self.prefactor_eloss_mse = 1.
        self.prefactor_symm = 1.

    def simplex_params(self, cls_expanded_simplex=False, time_scale=5):
        self.cls_expanded_simplex = cls_expanded_simplex
        self.time_scale = time_scale
        self.alpha_max = 8
        self.num_integration_steps = 20
        self.flow_temp = 1.
        self.allow_nan_cfactor = True

loss_mode = ["RC", "Energy"]
print("extra loss::", loss_mode)

hparams = Hyperparams(clean_data=False, num_cnn_stacks=3, hidden_dim=int(128), model="CNN2D", mode=loss_mode)
hparams.simplex_params()

from lightning_modules.simplex_module import simplexModule
model = simplexModule(channels, num_cls=72, num_e=32, hyperparams=hparams)



# import wandb
# wandb.init(
#     entity="anonymized",
#     settings=wandb.Settings(start_method="fork"),
#     project="betawolf",
#     name=args.run_name,
#     config=args,
# )
# from lightning.pytorch.loggers import WandbLogger
# 
# 
# wandb_logger = WandbLogger(project="my-project")
# wandb_logger.watch(model, log="all")

trainer = pl.Trainer(
    devices=1,
    default_root_dir=os.environ["MODEL_DIR"],
    accelerator="gpu" if torch.cuda.is_available() else 'auto',
    max_steps=max_steps,
    max_epochs=max_epochs,
    num_sanity_val_steps=0,
    limit_train_batches=limit_train_batches,
    limit_val_batches=limit_val_batches,
    enable_progress_bar=not (wandb) or os.getlogin() == 'ping-tuo',
    callbacks=[
        ModelCheckpoint(
            dirpath=os.environ["MODEL_DIR"],
            filename='model-{epoch:02d}-{train_loss:.2f}',
            save_top_k=-1,  # Save the top 3 models
            monitor='train_loss',  # Monitor validation loss
            mode='min',  # Minimize validation loss
            save_on_train_epoch_end=True
        )
    ],
    check_val_every_n_epoch=check_val_every_n_epoch,
    val_check_interval=val_check_interval,
    log_every_n_steps=1,
    precision="16-mixed"
)


if stage == "train":
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt)
else:
    trainer.validate(model, val_loader, ckpt_path=ckpt)
