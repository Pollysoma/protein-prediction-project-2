# embedding.py
# CLI to embed protein sequences from a FASTA file using ESM2
# - Accepts FASTA path
# - Optional savepath to write embeddings as .safetensors
# - Optional n to limit number of proteins (0 = all)
# - Reads and returns (id, seq) pairs from FASTA (IDs from ">" lines)
# - Batches inference and mean-pools over residues (excludes [CLS]/[EOS])

from __future__ import annotations

import argparse
import json
import os
import logging
from typing import Iterable, List, Tuple
from collections import Counter

import torch
from safetensors import safe_open
from safetensors.torch import save_file as safetensors_save
from tqdm import tqdm
import csv
from transformers import AutoModel, AutoTokenizer


UNKNOW_FRAGS_THRESHOLD = 0
MAX_SEQ_LENGTH = 1022

ESM2_SIZE_TO_MODEL = {
    "8M": "facebook/esm2_t6_8M_UR50D",
    "35M": "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
    "3B": "facebook/esm2_t36_3B_UR50D",
    "15B": "facebook/esm2_t48_15B_UR50D",
}


def read_fasta(path: str, n: int = 0) -> List[Tuple[str, str]]:
    """Parse FASTA file and return up to n (id, seq) pairs.

    - ID is the first token after '>' up to whitespace
    - Supports multi-line sequences
    - n == 0 means read all records
    """
    records: List[Tuple[str, str]] = []
    cur_id: str | None = None
    cur_seq_parts: List[str] = []

    def flush_record():
        nonlocal records, cur_id, cur_seq_parts
        if cur_id is not None:
            seq = "".join(cur_seq_parts).strip()
            if seq:
                records.append((cur_id, seq))
        cur_id = None
        cur_seq_parts = []

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                # save previous
                flush_record()
                # new id: first token after '>' up to whitespace
                header = line[1:].strip()
                cur_id = header.split()[0] if header else ""
                continue
            # sequence line
            cur_seq_parts.append(line)
            # we do not limit within a running record; handled after flush

    # flush last record
    flush_record()

    if n and n > 0:
        records = records[:n]

    return records

def read_tsv(path: str, n: int = 0) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []

    with open(path) as tsvfile:
        tsv_reader = csv.reader(tsvfile, delimiter="\t", quotechar='"')
        next(tsv_reader) # skip the header
        for row in tsv_reader:
            records.append((row[0], row[-1]))

            if n and len(records) >= n:
                break

    return records

def embed_batch(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    seqs: List[str],
    device: str,
) -> torch.Tensor:
    """Embed a batch of sequences -> tensor [B, H]."""
    enc = tokenizer(
        seqs,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
        truncation=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        #TODO another method instead of averaging all fragment embeddings is to just take the [CLS] embedding
        out = model(**enc)
        reps = out.last_hidden_state  # [B, T, H], includes [CLS] and [EOS]
        residue_reps = reps[:, 1:-1, :]  # drop [CLS] and [EOS]
        residue_mask = enc["attention_mask"][:, 1:-1].unsqueeze(-1)  # [B, L, 1]

        residue_reps = residue_reps * residue_mask
        lengths = residue_mask.sum(dim=1).clamp(min=1)  # [B, 1]
        embeddings = residue_reps.sum(dim=1, dtype=torch.float32) / lengths  # [B, H]
        return embeddings.cpu()


def batched(iterable: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def save_embeddings_safetensors(path: str, embeddings: torch.Tensor, ids: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    metadata = {"ids_json": json.dumps(ids)}
    tensors = {"embeddings": embeddings}
    safetensors_save(tensors, path, metadata=metadata)


def load_embeddings_safetensors(path: str) -> Tuple[torch.Tensor, List[str]]:
    """Load embeddings tensor and IDs from a .safetensors file.

    Returns (embeddings [B, H], ids list[str]).
    """
    with safe_open(path, framework="pt", device="cpu") as f:
        emb = f.get_tensor("embeddings")
        meta = f.metadata() or {}
    ids = json.loads(meta.get("ids_json", "[]"))
    return emb, ids


def resolve_device(arg_device: str | None) -> Tuple[str, torch.dtype]:
    if arg_device is None or arg_device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():  # For Apple Silicon
            device = "mps"
        else:
            device = "cpu"
    else:
        device = arg_device

    if device == "cuda":
        if not torch.cuda.is_available():
            logging.warning("CUDA requested but not available; falling back to CPU.")
            return "cpu", torch.float32

        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            logging.warning("Hardware with compute capability >= 8.0 detected, using bfloat16.")
            return "cuda", torch.bfloat16
        else:
            logging.warning("Hardware with compute capability < 8.0 detected (e.g., T4), using float16.")
            return "cuda", torch.float16

    return device, torch.float32


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="Embed protein sequences from FASTA/tsv using ESM2")
    parser.add_argument("input_path", type=str, help="Path to FASTA or tsv file")
    parser.add_argument(
        "--savepath",
        type=str,
        default=None,
        help="Optional path to write embeddings as .safetensors",
    )
    parser.add_argument(
        "--modelsize",
        type=str,
        choices=['8M', '35M', '150M', '650M', '3B', '15B'],
        default="35M",
        help="Choice of ESM2 model size",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=0,
        help="Number of proteins to process (0 = all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run on (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--flash-attention",
        action="store_true",
        help="Use optimized flash attention implementation"
    )
    args = parser.parse_args()

    if args.n < 0:
        raise SystemExit("--n must be >= 0")

    device, dtype = resolve_device(args.device)
    logging.info(f"Using device: {device} with dtype: {dtype}")

    ### read input file ###

    if args.input_path.endswith(".tsv"):
        records = read_tsv(args.input_path, n=args.n)
    elif args.input_path.endswith(".fasta"):
        records = read_fasta(args.input_path, n=args.n)
    else:
        raise SystemExit(f"Invalid input file type")

    if not records:
        raise SystemExit("No sequences found in the input file.")

    logging.info(f"Read {len(records)} sequences")

    ### clean data ###
    pre_cleaning_size = len(records)

    # truncate sequence length to MAX_SEQ_LENGTH and remove outlines with many unknown fragments
    records = [(rid, seq[:min(len(seq), MAX_SEQ_LENGTH)]) for rid, seq in records if
               Counter(seq).get("X", 0) <= UNKNOW_FRAGS_THRESHOLD]


    logging.info(f"Removed {pre_cleaning_size - len(records)} sequences during data cleaning")

    ids = [rid for rid, _ in records]
    seqs = [seq for _, seq in records]

    model_name = ESM2_SIZE_TO_MODEL[args.modelsize]

    # Load model/tokenizer once
    logging.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.flash_attention:
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"
    logging.info(f"Using attention implementation: {attn_implementation}")

    # Load model with half-precision and Flash Attention 2 if available
    model = AutoModel.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation=attn_implementation,
    ).to(device)

    try:
        model = torch.compile(model)
        logging.info("Model compiled successfully with torch.compile()")
    except Exception as e:
        logging.warning(f"torch.compile() failed: {e}. Running without compilation.")

    model.eval()

    # Run in batches to avoid OOM
    embeds: List[torch.Tensor] = []

    with tqdm(total=len(records), desc="Embedding sequences") as progress_bar:
        for batch in batched(seqs, args.batch_size):
            emb = embed_batch(model, tokenizer, batch, device)
            embeds.append(emb)
            progress_bar.update(len(batch))

    embeddings = torch.cat(embeds, dim=0)  # [B, H]

    # Output summary
    B, H = embeddings.shape
    logging.info(f"Embedded {B} sequences; embedding dim: {H}")
    # Show first 8 dims of first embedding
    first_preview = embeddings[0, : min(8, H)].tolist()
    logging.debug(f"First embedding preview (first 8 dims): {first_preview}")

    # Optional save
    if args.savepath:
        if not args.savepath.endswith(".safetensors"):
            logging.warning("savepath does not end with .safetensors; writing anyway.")
        save_embeddings_safetensors(args.savepath, embeddings, ids)
        logging.info(f"Saved embeddings to {args.savepath}")


if __name__ == "__main__":
    main()