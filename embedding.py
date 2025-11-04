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
from typing import Iterable, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file as safetensors_save
from transformers import AutoModel, AutoTokenizer


ESM2_SIZE_TO_MODEL = {
    "8M": "esm2_t6_8M_UR50D",
    "35M": "esm2_t12_35M_UR50D",
    "150M": "esm2_t30_150M_UR50D",
    "650M": "esm2_t33_650M_UR50D",
    "3B": "esm2_t36_3B_UR50D",
    "15B": "esm2_t48_15B_UR50D",
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
        out = model(**enc)
        reps = out.last_hidden_state  # [B, T, H], includes [CLS] and [EOS]
        residue_reps = reps[:, 1:-1, :]  # drop [CLS] and [EOS]
        residue_mask = enc["attention_mask"][:, 1:-1].unsqueeze(-1)  # [B, L, 1]
        residue_reps = residue_reps * residue_mask
        lengths = residue_mask.sum(dim=1).clamp(min=1)  # [B, 1]
        embeddings = residue_reps.sum(dim=1) / lengths  # [B, H]
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


def resolve_device(arg_device: str | None) -> str:
    if arg_device is None or arg_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if arg_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return arg_device


def main():
    parser = argparse.ArgumentParser(description="Embed protein sequences from FASTA using ESM2")
    parser.add_argument("fasta_path", type=str, help="Path to FASTA file")
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
    args = parser.parse_args()

    if args.n < 0:
        raise SystemExit("--n must be >= 0")

    device = resolve_device(args.device)

    records = read_fasta(args.fasta_path, n=args.n)
    if not records:
        raise SystemExit("No sequences found in the FASTA file.")

    ids = [rid for rid, _ in records]
    seqs = [seq for _, seq in records]


    model_name = ESM2_SIZE_TO_MODEL[args.modelsize]

    # Load model/tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    # Run in batches to avoid OOM
    embeds: List[torch.Tensor] = []
    for batch in batched(seqs, args.batch_size):
        emb = embed_batch(model, tokenizer, batch, device)
        embeds.append(emb)
    embeddings = torch.cat(embeds, dim=0)  # [B, H]

    # Output summary
    B, H = embeddings.shape
    print(f"Embedded {B} sequences; embedding dim: {H}")
    # Show first 8 dims of first embedding
    first_preview = embeddings[0, : min(8, H)].tolist()
    print("First embedding dims:", first_preview)

    # Optional save
    if args.savepath:
        if not args.savepath.endswith(".safetensors"):
            print("Warning: savepath does not end with .safetensors; writing anyway.")
        save_embeddings_safetensors(args.savepath, embeddings, ids)
        print(f"Saved embeddings to {args.savepath}")


if __name__ == "__main__":
    main()
