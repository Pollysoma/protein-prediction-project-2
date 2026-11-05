"""Per-protein ProtT5 embeddings for a FASTA file.

Frozen ProtT5-XL-U50 encoder in half precision; one vector per protein is the
mean over the last-layer residue embeddings (the final </s> token is excluded).
Same preprocessing as Rostlab/EAT: rare amino acids U, Z, O, B become X and
residues are separated by spaces before tokenising.

Output: HDF5 file with one float32 dataset of shape (1024,) per FASTA id.
"""

import argparse
import re
import time

import h5py
import torch
from transformers import T5EncoderModel, T5Tokenizer

MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"
REVISION = "94a6abc029ae13029317b140b7424e012bf8dfbf"


def read_fasta(path):
    seqs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                seq_id = line[1:].split()[0]
                seqs[seq_id] = []
            elif line:
                seqs[seq_id].append(line)
    return {seq_id: "".join(parts).upper() for seq_id, parts in seqs.items()}


def batches(seqs, max_residues):
    """Group sequences (longest first) so that each batch has at most max_residues."""
    batch, n_res = [], 0
    for seq_id, seq in sorted(seqs.items(), key=lambda kv: len(kv[1]), reverse=True):
        if batch and n_res + len(seq) > max_residues:
            yield batch
            batch, n_res = [], 0
        batch.append((seq_id, seq))
        n_res += len(seq)
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta")
    parser.add_argument("out_h5")
    parser.add_argument("--max-residues", type=int, default=8000)
    args = parser.parse_args()

    device = torch.device("cuda")
    tokenizer = T5Tokenizer.from_pretrained(MODEL, revision=REVISION, do_lower_case=False, legacy=True)
    model = T5EncoderModel.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float16)
    model = model.to(device).eval()

    seqs = read_fasta(args.fasta)
    start = time.time()
    with h5py.File(args.out_h5, "w") as out, torch.no_grad():
        for batch in batches(seqs, args.max_residues):
            ids = [seq_id for seq_id, _ in batch]
            lengths = [len(seq) for _, seq in batch]
            spaced = [" ".join(re.sub(r"[UZOB]", "X", seq)) for _, seq in batch]
            tokens = tokenizer(spaced, add_special_tokens=True, padding="longest", return_tensors="pt").to(device)
            hidden = model(**tokens).last_hidden_state
            for i, seq_id in enumerate(ids):
                emb = hidden[i, : lengths[i]].float().mean(dim=0)
                out.create_dataset(seq_id, data=emb.cpu().numpy())
    print(f"{len(seqs)} proteins embedded in {time.time() - start:.0f} s -> {args.out_h5}")


if __name__ == "__main__":
    main()
