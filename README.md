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
|   |-- raw/                Pre-partitioned RNA/ATAC data and biological priors
|   `-- processed/          Model-ready train/test data generated locally
|-- outputs/                Checkpoints, training histories, and predictions
|-- scPriorDiff/            Python source code
|-- scripts/                Data preparation, training, translation, and utilities
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

### 1. Obtain the repository and sample data

Download the four pre-partitioned `.h5ad` files from Google Drive:

[Download the scPriorDiff sample dataset](https://drive.google.com/drive/folders/1-goYf0v7SGT2ZD0EquxfnIpJISUWK74w?usp=sharing)

Copy the downloaded files into `data/raw/` under the repository root. Keep the file
names unchanged because they match the default configuration.

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

| Dependency | Reference version | Purpose |
| --- | ---: | --- |
| `anndata` | 0.11.4 | Reading and writing annotated single-cell matrices |
| `scanpy` | 1.11.1 | RNA preprocessing and feature selection |
| `h5py` | 3.9.0 | HDF5-backed `.h5ad` support |
| `numpy` | 1.24.3 | Dense numerical arrays and preprocessing |
| `pandas` | 2.2.3 | Cell, feature, and output tables |
| `scikit-learn` | 1.5.2 | PCA and clustering utilities |
| `scipy` | 1.15.3 | Sparse matrix storage and operations |
| `einops` | 0.8.1 | Tensor rearrangement in the diffusion network |
| `PyYAML` | 6.0.1 | Experiment configuration files |
| `torch` | 2.2.2 | Model definition, training, and inference |

## Sample dataset

The sample dataset is a paired scCAT-seq CellLine dataset distributed as four files
with a predefined training set and test set:

[Download the scPriorDiff sample dataset](https://drive.google.com/drive/folders/1-goYf0v7SGT2ZD0EquxfnIpJISUWK74w?usp=sharing)

After downloading, place the files under `data/raw/` as follows:

```text
scPriorDiff/
|-- data/
|   |-- raw/
|   |   |-- Paired_RNA_train.h5ad
|   |   |-- Paired_RNA_test.h5ad
|   |   |-- Paired_ATAC_train.h5ad
|   |   |-- Paired_ATAC_test.h5ad
|   |   |-- gact.csv
|   |   `-- priors/
|   |       |-- tf_edges.csv.gz
|   |       `-- chromosome_edges.csv.gz
|   `-- processed/          Created by scripts/prepare_data.py
|-- configs/
|   `-- cellline.yaml
`-- README.md
```

The four `.h5ad` files are downloaded from Google Drive. The proximal gene-peak table
and graph-prior files are provided with the repository.

### Input description

| Input | Required content |
| --- | --- |
| `Paired_RNA_train.h5ad` | Training cells by genes RNA matrix. Cell identifiers are stored in `obs_names`; gene identifiers are stored in `var_names`. |
| `Paired_RNA_test.h5ad` | Test cells by genes RNA matrix with the same gene features as the training RNA file. Its `obs` must contain `cell_type` as test-cell metadata. |
| `Paired_ATAC_train.h5ad` | Training cells by peaks ATAC matrix paired with the RNA training file. Its `var` must contain a `chrom` column. |
| `Paired_ATAC_test.h5ad` | Test cells by peaks ATAC matrix paired with the RNA test file and containing the same peak features as the training ATAC file. Its `var` must contain a `chrom` column. |
| `gact.csv` | Proximal gene-peak pairs with columns `peak` and `gene.name`. |
| `tf_edges.csv.gz` | Transcription-factor prior with columns `source` and `target`. |
| `chromosome_edges.csv.gz` | Chromosome-neighbor prior with columns `source`, `target`, and `distance`. |

Within each partition, RNA and ATAC files must contain the same unique cell identifiers.
The training and test cell identifiers must not overlap. RNA train/test files must have
the same genes, and ATAC train/test files must have the same peaks. The preparation
script verifies these conditions and aligns feature and cell ordering automatically.

Preprocessing parameters and feature selection are fitted using only the training
data. The same selected features and transformations are then applied to the test data.
The test data are not used by either training stage.

## Configuration

The default experiment is defined in [`configs/cellline.yaml`](configs/cellline.yaml).
It specifies the four input file names, input and output directories, preprocessing
thresholds, model dimensions, training parameters, diffusion settings, and random
seed. Paths in the configuration are interpreted relative to the repository root.

For a custom dataset, copy the configuration file, update the data paths and relevant
parameters, and pass it to every command using `--config`.

## Run scPriorDiff on the sample dataset

Run all commands from the repository root. The following example performs the
`RNA-ATAC` task. Replace `RNA-ATAC` with `ATAC-RNA` for the reverse task.

### 1. Prepare the data

This step reads the predefined training and test matrices together with the biological
priors. It validates pairing, fits preprocessing on the training set, applies the same
transformations to the test set, and writes model-ready files to `data/processed/`:

```bash
python scripts/prepare_data.py --config configs/cellline.yaml
```

The main preparation outputs are sparse RNA/ATAC matrices, graph-level matrices,
feature tables, `partitions.npz`, `graph.npz`, `cells.csv`, and `manifest.json`.

### 2. Train the representation model

```bash
python scripts/train_stage1.py --task RNA-ATAC --config configs/cellline.yaml
```

This command trains only on the predefined training cells and creates
`outputs/RNA-ATAC/stage1.pt` and `outputs/RNA-ATAC/stage1_history.csv`.

### 3. Train the conditional diffusion model

```bash
python scripts/train_diffusion.py --task RNA-ATAC --config configs/cellline.yaml
```

This command uses the trained representation model and the same predefined training
cells. It writes `diffusion.pt` and `diffusion_history.csv` to `outputs/RNA-ATAC/`.

### 4. Generate predictions for the test set

```bash
python scripts/translate.py --task RNA-ATAC --config configs/cellline.yaml
```

For `RNA-ATAC`, the output is `predicted_atac.h5ad`. For `ATAC-RNA`, the output is
`predicted_rna.h5ad`. Rows correspond to the predefined test cells, columns correspond
to features in the predicted modality, `obs` contains test-cell metadata, and `var`
contains output feature metadata.

To force CPU execution, append `--device cpu` to the two training commands and the
translation command. Runtime and memory use depend on the selected task,
hardware, and configuration values.

## Expected outputs

After a complete `RNA-ATAC` run, the output directory is:

```text
outputs/
`-- RNA-ATAC/
    |-- config.yaml
    |-- stage1.pt
    |-- stage1_history.csv
    |-- diffusion.pt
    |-- diffusion_history.csv
    `-- predicted_atac.h5ad
```

The `.pt` files are model checkpoints. The history files contain epoch-level training
losses. The prediction `.h5ad` file contains the generated target-modality matrix for
the predefined test cells, together with their observation and feature metadata.

## Troubleshooting

- If a sample-data file is missing, download it from the Google Drive link and confirm
  that all four `.h5ad` files are present under `data/raw/` with unchanged names.
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
