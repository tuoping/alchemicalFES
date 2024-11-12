import sys
from argparse import ArgumentParser
import subprocess, os
from datetime import datetime




def parse_train_args():
    parser = ArgumentParser()
    
    parser.add_argument("--validation_batch_size", type=int, default=8192)
    parser.add_argument("--dump_freq", type=int, default=1)

    # Run settings
    parser.add_argument("--ckpt_epoch", type=int, default=None)
    parser.add_argument("--workdir_descriptor", type=str, default="")
    parser.add_argument("--modeldir_temperature", type=str, default="3.2")
    parser.add_argument("--modeldir_type", type=str, default="clsfreeG")
    parser.add_argument("--validation_lattice_size", type=int, default=6)
    parser.add_argument("--validation_temperature", type=float)

    parser.add_argument("--uncond_model_ckpt", type=None)

    parser.add_argument("--alpha_max", type=int, default=10)
    parser.add_argument("--num_integration_steps", type=int, default=80)
    parser.add_argument("--shuffle_cls_freq", type=float, default=0.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)

    parser.add_argument("--cls_guidance", action="store_true")
    parser.add_argument("--cls_guidance_dataset_dir", type=str, default=None)

    parser.add_argument("--clsfree_guidance", action="store_true")
    parser.add_argument("--probability_tilt", action="store_true")
    parser.add_argument("--probability_tilt_scheduled", action="store_true")
    parser.add_argument("--clsfree_guidance_dataset", action='store_true')
    parser.add_argument("--clsfree_guidance_dataset_file", type=str, default=None)
    parser.add_argument("--clsfree_guidance_dataset_lattice_size", type=int, default=6)
    args = parser.parse_args()
    
    os.environ["MODEL_DIR"]=f"/nfs/scistore23/chenggrp/ptuo/NeuralRG/alchemicalFES/logs-dir-ising/latt6x6T{args.modeldir_temperature}/kernel3x3_timeembed_symmetrized/{args.modeldir_type}"
    os.environ["work_dir"]=os.path.join(os.environ["MODEL_DIR"], f"val_baseline_latt{args.validation_lattice_size}x{args.validation_lattice_size}/epoch{args.ckpt_epoch}_{args.workdir_descriptor}")

    import glob
    if args.ckpt_epoch is not None:
        args.ckpt = glob.glob(os.path.join(os.environ["MODEL_DIR"], f"model-epoch={args.ckpt_epoch}-train_loss=*"))[0]
    else:
        args.ckpt = None

    args.dataset_dir = f"data/ising-latt{args.validation_lattice_size}x{args.validation_lattice_size}-T{args.validation_temperature}"

    args.scale_magn = int((args.validation_lattice_size/6)**2)
    if args.cls_guidance:
        assert args.cls_guidance_dataset_dir is not None
    return args
