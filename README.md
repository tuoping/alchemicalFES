# Alchemical FES
Modified from https://github.com/HannesStark/dirichlet-flow-matching

## Installation
Conda environment
```bash
conda create -c conda-forge -n seq python=3.9
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
pip install torch_geometric jupyterlab gpustat pyyaml wandb biopython spyrmsd einops biopandas plotly seaborn prody tqdm lightning imageio tmtools "fair-esm[esmfold]" e3nn
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cu113.htm

# The libraries below are required for the promoter design experiments
git clone https://github.com/kathyxchen/selene.git
cd selene
git checkout custom_target_support
python setup.py build_ext --inplace
python setup.py install

pip install pyBigWig pytabix cooler pyranges biopython cooltools
```
## Ising model
Validate
```bash
python -u val_simplex_ising.py --dump_freq 4 --ckpt_epoch 162 --modeldir_type clsfreeG/Eloss/guidanceM/test2-mixTdata --workdir_descriptor T2.0_Int80Amax10 --uncond_model_ckpt logs-dir-ising/latt6x6T3.2/kernel3x3_timeembed_symmetrized/eloss_uncond/addmseloss/model-epoch=109-train_loss=3.79.ckpt --clsfree_guidance --probability_tilt --clsfree_guidance_dataset --validation_temperature 2.0
```
