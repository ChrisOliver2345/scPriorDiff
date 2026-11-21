# scPriorDiff

This is the official repository for scPriorDiff. scPriorDiff is a conditional latent
diffusion model guided by biological network priors for cross-modal translation between
single-cell transcriptomic and chromatin accessibility profiles. It supports two
translation tasks:

- `RNA-ATAC`: predict chromatin accessibility from gene expression.
- `ATAC-RNA`: predict gene expression from chromatin accessibility.

## Repository layout

```text
scPriorDiff/
|-- configs/
|   `-- cellline.yaml       Example experiment configuration
|-- data/
|   |-- raw/                RNA/ATAC data and biological priors
|   `-- processed/          Model-ready train/test data generated locally
|-- outputs/                Checkpoints, training histories, and predictions
|-- scPriorDiff/            Python source code
|-- requirements.txt        Complete list of direct Python dependencies
`-- README.md               Installation and usage documentation
```

## Installation

A separate virtual environment is recommended for installing and running scPriorDiff.

### Reference environment

- 64-bit Python 3.11 (Python 3.10 is also expected to be compatible)
- `pip` 23 or newer
- Linux or Windows
- NVIDIA CUDA-capable GPU recommended for training
- CPU execution supported, but potentially much slower

The exact direct dependency versions used for the reference environment are pinned
in [`requirements.txt`](requirements.txt). Dependencies required transitively by
these packages are resolved automatically by `pip`.

### 1. Prepare the input files

Place the four RNA and ATAC `.h5ad` files in `data/raw/` under
the repository root. Use the filenames listed under [Input file layout](#input-file-layout)
for the default configuration.

### 2. Create and activate a virtual environment

On Linux or macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install the Python dependencies

For CPU execution:

```bash
python -m pip install -r requirements.txt
```

For an NVIDIA GPU, first install the CUDA-enabled build of `torch==2.2.2` appropriate
for the operating system and driver. The authoritative commands are available on the
[PyTorch previous versions page](https://pytorch.org/get-started/previous-versions/).
For example, the CUDA 12.1 wheel can be installed with:

```bash
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

The second command installs the remaining dependencies and retains the compatible
PyTorch wheel installed first.

### 4. Verify the installation

Run the following commands from the repository root:

```bash
python -c "import anndata, einops, h5py, numpy, pandas, scanpy, scipy, sklearn, torch, yaml; print('All dependencies imported successfully.')"
python -c "import scPriorDiff, torch; print('scPriorDiff import: OK'); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Dependencies

All packages imported directly by scPriorDiff or its command-line workflow are listed
in `requirements.txt`.

| Dependency       | Reference version | Purpose                                            |
| ---------------- | ----------------: | -------------------------------------------------- |
| `anndata`      |            0.11.4 | Reading and writing annotated single-cell matrices |
| `scanpy`       |            1.11.1 | RNA preprocessing and feature selection            |
| `h5py`         |             3.9.0 | HDF5-backed `.h5ad` support                      |
| `numpy`        |            1.24.3 | Dense numerical arrays and preprocessing           |
| `pandas`       |             2.2.3 | Cell, feature, and output tables                   |
| `scikit-learn` |             1.5.2 | PCA and clustering utilities                       |
| `scipy`        |            1.15.3 | Sparse matrix storage and operations               |
| `einops`       |             0.8.1 | Tensor rearrangement in the diffusion network      |
| `PyYAML`       |             6.0.1 | Experiment configuration files                     |
| `torch`        |             2.2.2 | Model definition, training, and inference          |

## Input file layout

The default configuration uses the four RNA and ATAC files shown below.

```text
scPriorDiff/
|-- data/
|   |-- raw/
|   |   |-- Paired_RNA_train.h5ad
|   |   |-- Paired_RNA_test.h5ad
|   |   |-- Paired_ATAC_train.h5ad
|   |   `-- Paired_ATAC_test.h5ad
|   `-- processed/
|-- configs/
|   `-- cellline.yaml
`-- README.md
```

## Configuration

The default experiment is defined in [`configs/cellline.yaml`](configs/cellline.yaml).
It specifies the four input file names, input and output directories, preprocessing
thresholds, model dimensions, training parameters, diffusion settings, and random
seed. Paths in the configuration are interpreted relative to the repository root.

For a custom dataset, copy the configuration file, update the data paths and relevant
parameters, and pass it to every command using `--config`.

## Troubleshooting

- If an input file is missing, check that all four `.h5ad` files are in
  `data/raw/` and that their names match the configuration.
- If data preparation reports mismatched cells or features, verify that corresponding
  RNA and ATAC files were generated from the same paired partition and that train/test
  modality files use identical feature identifiers.
- If Python cannot import `scPriorDiff`, run the command from the repository root and
  confirm that the intended virtual environment is active.
- If `torch.cuda.is_available()` returns `False`, confirm that a CUDA-enabled PyTorch
  wheel and a compatible NVIDIA driver are installed. CPU execution remains available
  through `--device cpu`.
- If GPU memory is insufficient, reduce the applicable batch size in the configuration
  file and use a clean output directory for the new run.
