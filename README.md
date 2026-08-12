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
  attention_unet.py  Attention-based 1D U-Net used as the diffusion denoiser
  completion.py      Prior-guided cross-attention for target-node completion
  config.py          Task definitions and experiment configuration loading
  data.py            Processed-data loading, graph loading, and data loaders
  diffusion.py       Diffusion training objective and reverse sampling process
  graph.py           Heterogeneous graph encoding and message-passing layers
  model.py           Stage-one model assembly and cross-modal latent encoding
  modules.py         RNA and chromosome-aware ATAC encoders and decoders
  training.py        Training loops, latent extraction, and checkpoint helpers
requirements.txt     Python dependencies
README.md            Project overview, package layout, and installation
```

## Installation

Python 3.10 or newer and a CUDA-capable PyTorch installation are recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```
