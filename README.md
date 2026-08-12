# scPriorDiff

scPriorDiff is a conditional latent diffusion model guided by biological network
priors for cross-modal translation between single-cell transcriptomic and chromatin
accessibility profiles. It integrates diverse biological networks into a multimodal
heterogeneous graph, using GNN-based message passing and prior-regularized cross-modal
completion to capture intra- and cross-modal relationships and guide target-modality
generation.

## Project layout

```text
scPriorDiff/
  __init__.py         Package exports
  attention_unet.py  Diffusion denoiser
  completion.py      Cross-modal completion
  config.py          Configuration loading
  data.py            Data loading
  diffusion.py       Diffusion process
  graph.py           Heterogeneous graph encoder
  model.py           scPriorDiff model
  modules.py         Modality encoders and decoders
  training.py        Training utilities
requirements.txt     Python dependencies
README.md            Project documentation
```

## Installation

Python 3.10 or newer and a CUDA-capable PyTorch installation are recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```
