
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils.parsing import parse_train_args
args = parse_train_args()

import torch
import os,sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"


stage = "val"
channels = 2
L = args.validation_lattice_size
seq_len = args.validation_lattice_size**2
seq_dim = (args.validation_lattice_size, args.validation_lattice_size)
ckpt = args.ckpt
if stage == "train":
    batch_size = 1024
    if ckpt is not None: 
        print("Starting from ckpt:: ", ckpt)
elif stage == "val":
    batch_size = args.validation_batch_size
    if ckpt is None: 
        raise Exception("ERROR:: ckpt not initiated")
    print("Validating with ckpt::", ckpt)
else:
    raise Exception("Unrecognized stage")


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
    inference_mode=False,
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
            save_top_k=3,  # Save the top 3 models
            monitor='train_loss',  # Monitor validation loss
            mode='min',  # Minimize validation loss
            every_n_train_steps=1000,  # Checkpoint every 1000 training steps
        )
    ],
    check_val_every_n_epoch=check_val_every_n_epoch,
    val_check_interval=val_check_interval,
    log_every_n_steps=1,
    precision="32"
)

class dataset_params():
    def __init__(self, toy_seq_len, toy_seq_dim, toy_simplex_dim, dataset_dir, scale_magn = 1):
        self.toy_seq_len = toy_seq_len
        self.toy_seq_dim = toy_seq_dim
        self.toy_simplex_dim = toy_simplex_dim
        self.dataset_dir = dataset_dir
        self.cls_ckpt = None
        self.scale_magn = scale_magn
        if stage == "val":
            self.subset_size = batch_size
        else:
            self.subset_size = None
        
dparams = dataset_params(seq_len, seq_dim, channels, args.dataset_dir, scale_magn = args.scale_magn)

from utils.dataset import IsingDataset

dparams.dataset_dir = os.path.join(args.dataset_dir, "val")

train_ds = IsingDataset(dparams)
val_ds = IsingDataset(dparams)
if not args.clsfree_guidance_dataset:
    val_ds.make_custom_target_class()
else:
    if args.clsfree_guidance_dataset_file is not None:
        val_ds.read_target_class(args.clsfree_guidance_dataset_file, seq_L=args.clsfree_guidance_dataset_lattice_size, scale_magn=int((args.clsfree_guidance_dataset_lattice_size/6)**2), subset_size=batch_size)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, drop_last=True)

class Hyperparams():
    def __init__(self, mode=None, hidden_dim=16, num_cnn_stacks=1, lr=5e-4, dropout=0.0, clean_data=False, model="MLP"):
        self.hidden_dim = hidden_dim
        self.kernel_size = 3
        self.padding = 1
        self.dropout = dropout

        self.cls_free_guidance = args.clsfree_guidance
        self.probability_tilt = args.probability_tilt
        self.probability_tilt_scheduled = args.probability_tilt_scheduled
        self.guidance_op = "energy-magnetization"
        self.uncond_model_ckpt = args.uncond_model_ckpt

        self.clean_data = clean_data
        self.num_cnn_stacks = num_cnn_stacks
        self.lr = lr
        self.wandb = False
        self.seq_dim = (args.validation_lattice_size,args.validation_lattice_size)
        self.channels = channels
        self.model = model
        self.mode = mode
        self.gamma_focal = 2.
        self.prefactor_CE = 1.
        if mode is not None and "RC" in mode:
            self.prefactor_RC = 1.
            self.prefactor_CE = 0.01
        if mode is not None and "Energy" in mode:
            self.prefactor_EKL = 1.
            self.prefactor_CE = 0.01
            self.EKLloss_temperature = 500
            self.prefactor_eloss_mse = 0.1
        self.prefactor_symm = 1.
    

    def simplex_params(self, time_scale=2):
        self.enforce_symm = False
        
        self.cls_free_noclass_ratio = 0.0
        self.time_scale = time_scale
        self.alpha_max = args.alpha_max
        self.num_integration_steps = args.num_integration_steps
        self.flow_temp = 1.0
        self.allow_nan_cfactor = True

        self.prior_pseudocount = 0.
        self.score_free_guidance = False
        self.guidance_scale = args.guidance_scale
        self.shuffle_cls_freq = args.shuffle_cls_freq
        
        self.cls_guidance = args.cls_guidance
        self.analytic_cls_score = True
        self.cls_expanded_simplex = False
        self.scale_cls_score = False

        self.dump_freq = args.dump_freq

loss_mode = ["Energy"]
print(">>> Using extra loss::", loss_mode)

hparams = Hyperparams(clean_data=False, num_cnn_stacks=3, hidden_dim=int(128), model="CNN2D", mode=loss_mode)
hparams.simplex_params()

from lightning_modules.simplex_module import simplexModule
if args.cls_guidance_dataset_dir is not None:
    toy_data_dir = args.cls_guidance_dataset_dir
    toy_dparams = dataset_params(6*6, (6,6), 2, toy_data_dir, scale_magn = args.scale_magn)
    toy_ds = IsingDataset(toy_dparams)
else:
    toy_ds = None

# model = simplexModule(channels, 72, 32, hyperparams=hparams, toy_data=toy_ds).load_from_checkpoint(ckpt, strict=False)

model = simplexModule.load_from_checkpoint(
    checkpoint_path=ckpt,
    strict=False,  # if needed
    hyperparams=hparams,
    # Provide any additional arguments if required
)

if stage == "train":
    trainer.fit(model, train_loader, val_loader)
else:
    trainer.validate(model, val_loader)
