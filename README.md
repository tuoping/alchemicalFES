# Alchemical FES

## Installation
Conda environment
```bash
conda create -c conda-forge -n seq python=3.9
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
pip install torch_geometric jupyterlab gpustat pyyaml wandb biopython spyrmsd einops biopandas plotly seaborn prody tqdm lightning imageio tmtools "fair-esm[esmfold]" e3nn
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cu113.htm

```
## Ising model
### Train
Conditional FM using Potential energy and Magnetization order parameters as input conditions
```bash
python -u src/Train-IsingModel-EMcondition.py
```

### Validate
This is the validation command for conditional model, conditioning on data of the 6x6 Ising model at $k_BT=2.0$.
```bash
python -u val_simplex_ising.py --dump_freq 4 --ckpt_epoch 162 --modeldir_type clsfreeG/Eloss/guidanceM/test2-mixTdata --workdir_descriptor T2.0_Int80Amax10 --uncond_model_ckpt logs-dir-ising/latt6x6T3.2/kernel3x3_timeembed_symmetrized/eloss_uncond/addmseloss/model-epoch=109-train_loss=3.79.ckpt --clsfree_guidance --probability_tilt --clsfree_guidance_dataset --validation_temperature 2.0
```
## Acknowledgements
Code development based on

[Dirichlet Flow Matching with Applications to DNA Sequence Design](https://github.com/HannesStark/dirichlet-flow-matching)
