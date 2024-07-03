
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import torch
import os,sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

os.environ["MODEL_DIR"]=f"logs-dir-ising/latt6x6_d/"
os.environ["work_dir"]=os.path.join(os.environ["MODEL_DIR"], f"val_baseline_latt6x6/epoch{sys.argv[1]}_sample{sys.argv[2]}")
dataset_dir = "ising-latt6x6-anneal"

stage = "val"
channels = 2
seq_len = 6*6
seq_dim = (6, 6)
ckpt = None
import glob
ckpt = glob.glob(os.path.join(os.environ["MODEL_DIR"], f"model-epoch={sys.argv[1]}-train_loss=*"))[0]

num_workers = 0
max_steps = 100000
max_epochs = 100000
limit_train_batches = None
if stage == "train":
    limit_val_batches = 0.0
else:
    limit_val_batches = 100
grad_clip = 1.
wandb = False
check_val_every_n_epoch = None
val_check_interval = None

trainer = pl.Trainer(
    devices=1, num_nodes=1,
    default_root_dir=os.environ["work_dir"],
    accelerator="gpu" if torch.cuda.is_available() else 'auto',
    max_steps=max_steps,
    max_epochs=max_epochs,
    num_sanity_val_steps=0,
    limit_train_batches=limit_train_batches,
    limit_val_batches=limit_val_batches,
    enable_progress_bar=not (wandb) or os.getlogin() == 'ping-tuo',
    gradient_clip_val=grad_clip,
    callbacks=[
        ModelCheckpoint(
            dirpath=os.environ["MODEL_DIR"],
            filename='model-{epoch:02d}-{train_loss:.2f}',
            save_top_k=-1,  # Save the top 3 models
            monitor='train_loss',  # Monitor validation loss
            mode='min',  # Minimize validation loss
            every_n_train_steps=5000,  # Checkpoint every 1000 training steps
        )
    ],
    check_val_every_n_epoch=check_val_every_n_epoch,
    val_check_interval=val_check_interval,
    log_every_n_steps=1,
    precision=16,
    strategy='ddp_find_unused_parameters_true'
)

class dataset_params():
    def __init__(self, toy_seq_len, toy_seq_dim, toy_simplex_dim, dataset_dir):
        self.toy_seq_len = toy_seq_len
        self.toy_seq_dim = toy_seq_dim
        self.toy_simplex_dim = toy_simplex_dim
        self.dataset_dir = dataset_dir
        self.t_max = 10
        self.t_min = 0.0001
        self.dataset_files = ["buffer_enhancelowTmaxT_ordered_dt0.1.npy", "t_enhancelowTmaxT_ordered_dt0.1.npy"]
        self.basis="simplex"
        self.subset = False
        
dparams = dataset_params(seq_len, seq_dim, channels, dataset_dir)

from utils.dataset import AlCuDataset, IsingDataset
# dparams.dataset_dir = dataset_dir
dparams.dataset_dir = os.path.join(dataset_dir, "val")
train_ds = IsingDataset(dparams)
dparams.dataset_dir = os.path.join(dataset_dir, "val")
val_ds = IsingDataset(dparams)

if stage == "train":
    batch_size = 1024
    if ckpt is not None: 
        print("Starting from ckpt:: ", ckpt)
elif stage == "val":
    batch_size = val_ds.all_t.shape[0]
    # batch_size = 2048
    if ckpt is None: 
        raise Exception("ERROR:: ckpt not initiated")
    print("Validating with ckpt::", ckpt)
else:
    raise Exception("Unrecognized stage")

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, drop_last=True)

class Hyperparams():
    def __init__(self, mode=None, hidden_dim=16, num_cnn_stacks=1, lr=5e-4, dropout=0.0, cls_free_guidance=False, clean_data=False, model="MLP"):
        self.hidden_dim = hidden_dim
        self.kernel_size = 3
        self.padding = 1
        self.dropout = dropout
        self.cls_free_guidance = cls_free_guidance
        self.clean_data = clean_data
        self.num_cnn_stacks = num_cnn_stacks
        self.lr = lr
        self.wandb = False
        self.seq_dim = seq_dim
        self.channels = channels
        self.model = model
        self.mode = mode
        if mode is not None and "RC" in "".join(mode):
            self.prefactor_RC = 1.
            self.tgrid_num_alpha=20
            self.tgrid_bandwidth=2.0
        if mode is not None and "multinomial" in mode:
            self.prefactor_M = 1.
        if mode is not None and "Energy" in mode:
            self.prefactor_E = 1.
        self.prefactor_CE = 0.05
        self.classifier = False
        if self.classifier:
            self.prefactor_E = 1.

        self.t_max = 10.
        self.t_min = 0.0001


    def simplex_params(self, cls_expanded_simplex=False, time_scale=2):
        self.cls_expanded_simplex = cls_expanded_simplex
        self.time_scale = time_scale
        self.alpha_max = 8
        self.num_integration_steps = 20
        self.flow_temp = 1.0
        self.allow_nan_cfactor = True
        self.prior_from_file=False

loss_mode = ["RC-t"]
print("extra loss::", loss_mode)
hparams = Hyperparams(clean_data=False, num_cnn_stacks=3, hidden_dim=int(128), model="CNN2D", mode=loss_mode)
hparams.simplex_params()


from lightning_modules.simplex_module import simplexModule
model = simplexModule(channels, num_cls=1, hyperparams=hparams)

if stage == "train":
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt)
else:
    trainer.validate(model, val_loader, ckpt_path=ckpt)
