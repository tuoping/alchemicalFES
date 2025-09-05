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
python -u val_simplex_ising.py --ckpt_epoch 209 --workdir_descriptor $workdir_descriptor --modeldir_temperature 3.2 --validation_lattice_size 24 --modeldir_type kernel3x3_timeembed_symmetrized/clsfreeG/ElossMloss/guidanceEM --validation_temperature 4.0 --shuffle_cls_freq 0.0 --guidance_scale $1 --uncond_model_ckpt $uncond_model_ckpt --clsfree_guidance  --num_integration_steps 640 --dump_freq 8 --probability_tilt 
```
## Acknowledgements
Code development based on

[Dirichlet Flow Matching with Applications to DNA Sequence Design](https://github.com/HannesStark/dirichlet-flow-matching)
