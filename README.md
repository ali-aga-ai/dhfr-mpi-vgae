# DHFR MPI-VGAE: Mutation-Aware Pathway Link Prediction

This project uses Graph Neural Networks (MPIVGAE) to predict protein-metabolite interactions in biological pathways and evaluate how mutations affect network connectivity. The primary use case is studying the Plasmodium folate biosynthesis pathway for drug resistance analysis.

## Project Overview

- **Goal**: Learn pathway structure using GNNs and predict mutation-induced network rewiring
- **Model**: MPIVGAE (Message Passing Interface - Variational Graph Autoencoder)
- **Current Status**: Pipeline operational for within-species and mutation experiments; cross-species generalization remains an open challenge

## Repository Structure (Useful files)

```
mpi-vgae/
├── notebooks/
│   ├── vgae_pipeline.py          # Main CLI tool for training & inference
│   ├── commented_run.ipynb       # Detailed walkthrough - START HERE for understanding
│   ├── mpi_vgae_run.ipynb        # General MPIVGAE execution
│
├── create_proteins/
│   ├── prepare_protein_fasta.py  # Fetch protein sequences from UniProt
│   ├── embed.py                  # Generate ESM embeddings from FASTA
│   ├── project_protein.py        # Dimensionality reduction (1280d → 1024d)
│   └── ecoli_protein_names.fasta # Sample protein sequence file
│
├── features/
│   ├── mpi_network/              # Network topology files (.pkl)
│   └── mpi_features/             # Node feature files (.pkl)
│
└── README.md                     # This file
```

## Quick Start

### Understanding the Codebase

**Start with**: `notebooks/commented_run.ipynb` - contains detailed comments explaining the full pipeline.

### Running the Main Pipeline

The primary tool is `vgae_pipeline.py` - a CLI for training and inference.

**Important**: This pipeline is designed for **within-species** or **original-to-mutated** experiments, NOT cross-species training/validation. Since certain nodes present in one, may not exist in another.

#### Example Usage

```bash
# On the server, use /usr/bin/python3
/usr/bin/python3 vgae_pipeline.py \
    --network features/mpi_network/mpi_Homo_sapiens.pkl \
    --train-features features/mpi_features/feature_df_Homo_sapiens.pkl \
    --inference-features features/mpi_features/feature_df_Homo_sapiens.pkl \
    --epochs 500 \
    --cutoff 0.67 \
    --output-prefix pipeline_test
```

#### CLI Arguments

```
Required:
  --network               Path to MPI network pickle file
  --train-features        Path to training node features pickle
  --inference-features    Path to inference node features pickle

Optional:
  --model-weights         Pretrained model checkpoint (skips training if provided)
  --save-weights          Path to save trained model weights
  --epochs                Number of training epochs (default: 500)
  --lr                    Learning rate (default: 0.005)
  --hidden1               Hidden layer 1 dimension (default: 32)
  --hidden2               Hidden layer 2 dimension (default: 16)
  --dropout               Dropout rate (default: 0.1)
  --cutoff                Prediction threshold (default: 0.67)
  --output-prefix         Prefix for output files (default: 'vgae')
```

#### Expected Outputs

- Confusion matrices (PDF format)
- Training/validation/inference metrics (precision, recall, AUC-ROC, AP)
- Loss curves
- Edge predictions

## Workflow: Creating Mutated Protein Features

To simulate mutations and generate new node features:

### 1. Prepare Protein Sequences

```bash
cd create_proteins
python prepare_protein_fasta.py
```

This script:
- Fetches protein sequences from UniProt using IDs from your dataset
- Outputs FASTA files for downstream embedding

### 2. Generate ESM Embeddings

```bash
python embed.py
```

This script:
- Reads FASTA files from step 1
- Generates 1280-dimensional ESM protein embeddings
- Outputs embedding vectors

### 3. Project to Lower Dimensions

```bash
python project_protein.py
```

This script:
- Projects 1280d protein embeddings to 1024d (to match metabolite dimensions)
- Currently uses random projection (can be changed to PCA if sample size allows)
- Outputs consistent-dimension feature files for MPIVGAE

### 4. Run MPIVGAE Pipeline

Use the projected features in `vgae_pipeline.py` as shown above.

## Data Structure

### Network Files (`features/mpi_network/`)

Pickle files containing graph structure:
- Node indices
- Edge lists (protein-metabolite interactions)
- Adjacency matrices

### Feature Files (`features/mpi_features/`)

Pickle files containing node embeddings:
- Protein vectors 
- Metabolite vectors 
- Node metadata (UniProt IDs, node types ETC)

## Environment Setup

The project requires:
- Python 3.x
- TensorFlow (with TF1 compatibility layer for legacy code)
- CUDA-enabled GPU (tested on NVIDIA RTX A6000)
- ESM protein language model
- Standard scientific Python stack (NumPy, Pandas, scikit-learn)

**Note**: Python versions and TensorFlow configurations are pre-configured on the server. Contact the original team if setting up a new environment.

## Key Findings and Known Issues

### What Works

- **Within-species prediction**: Strong performance
- **Mutation detection**: Pipeline successfully captures feature-driven edge changes
- **End-to-end workflow**: Sequence → Embedding → GNN → Predictions

### Known Limitations

1. **Cross-species generalization fails**: Models trained on one organism perform at random chance on others
2. **TensorFlow 1 legacy code**: Requires compatibility layer, prone to environment issues
3. **Dimensionality mismatch workaround**: Random projection used instead of PCA (insufficient samples)
4. **Heuristic node classification**: Proteins in "other" category may be missed in Homo sapiens dataset
5. **Mutation realism**: Current mutations are random and not biologically curated
6. **Small mutation dataset**: Limited validation data for mutated proteins

### Critical Notes for Future Work

- **Plasmodium dataset**: Not yet integrated; focus has been on Homo sapiens
- **Cross-species training**: Merged multi-species models did not improve generalization
- **Feature engineering**: Correlation matrices must be reconstructed carefully to match original methodology

## References

- Wang et al. (2023). "MPI-VGAE: protein–metabolite enzymatic reaction link learning by variational graph autoencoders". *Bioinformatics*.
- Original implementation: TensorFlow 1 framework for GNN-based link prediction

## Project Repository

Full code available at: [https://github.com/ali-aga-ai/dhfr-mpi-vgae](https://github.com/ali-aga-ai/dhfr-mpi-vgae)
