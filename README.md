# Alchemical FES
Modified from https://github.com/HannesStark/dirichlet-flow-matching

## Ising model
Validate
```bash
python -u val_simplex_ising.py --dump_freq 4 --ckpt_epoch 162 --modeldir_type clsfreeG/Eloss/guidanceM/test2-mixTdata --workdir_descriptor T2.0_Int80Amax10 --uncond_model_ckpt logs-dir-ising/latt6x6T3.2/kernel3x3_timeembed_symmetrized/eloss_uncond/addmseloss/model-epoch=109-train_loss=3.79.ckpt --clsfree_guidance --probability_tilt --clsfree_guidance_dataset --validation_temperature 2.0
```
