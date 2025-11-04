TODO:

## Dataset
- [Polly] Pretraining Dataset with UniProt API
- [Polly] Training Set Statistics

## Presentation
- [Everyone] Add picture to presentation

## Embeddings
8M, 35M, (150M) ESM-2 per Protein and per Token Embeddings
Safe tensor dataformat
- [Timon] New Proteins Embeddings
- [Timon] Pre Training Embeddings
- [Alexander] Dimensionality Reduction

## binary predictor
- [Egor] Baseline Architecture

## Literature Research
- [Aladdin] membership inference attacks on LLMs


# Usage of the project

## Prerequisites
To manage dependencies and python version use uv.
[uv install guide](https://docs.astral.sh/uv/getting-started/installation/)

### Installing dependencies
Simply run `uv sync` which handles installing all dependencies locked to specific versions using hashes and also handles the installation of the pinned python version.

### Adding dependencies
To add a new dependency run `uv add {dependency-name}`. This adds it to your environment as well as adding it to the uv.lock file ensuring everyone uses the same version of all dependencies.

## Embedding Script Usage
- Script: `embedding.py` (ESM2-based embeddings)
- Install deps: `pip install -e .` (or `uv pip install -e .`)
- Basic: `python embedding.py path/to/file.fasta`

### Possible flags:
- Save to safetensors: `--savepath outputs/embeddings.safetensors`
- Limit N sequences: `--n 100`
- Model size: `--modelsize 8M|35M|150M|650M|3B|15B` (default 35M)
- Device: `--device auto|cpu|cuda` (default `auto`)
- Batch size: `--batch-size 32` (default 32)

Notes
- FASTA headers' first token after `>` is used as the sequence ID; IDs are stored in the `.safetensors` metadata.
- Load helper:
  ```python
  from embedding import load_embeddings_safetensors
  emb, ids = load_embeddings_safetensors("outputs/embeddings.safetensors")
  ```

Example (project data)
- Embed all proteins in the provided dataset and save:
  ```bash
  python embedding.py data/newly_discovered_dissimilar_proteins_2024.fasta --savepath outputs/embedding-new-2.safetensors
  ```
- Load them back in Python:
  ```python
  from embedding import load_embeddings_safetensors
  emb, ids = load_embeddings_safetensors("outputs/embedding-new-2.safetensors")
  print(emb.shape, len(ids))
  ```
