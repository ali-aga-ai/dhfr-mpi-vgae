#!/usr/bin/env python3
import os, re, gzip, json, pickle, argparse
from pathlib import Path
from typing import Iterator, Tuple, List, Dict
import numpy as np
from tqdm import tqdm

import torch
# prefer the explicit ESM classes first; fall back to Auto*
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
try:
    from transformers import EsmModel, EsmForMaskedLM
except Exception:  # old transformers may not expose these symbols
    EsmModel = None
    EsmForMaskedLM = None
#  /srv/shared/esm2/esm2_t33_650M_UR50D.pt
HF_CACHE = Path("./hf_cache").resolve()
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)

AMINO = set("ACDEFGHIKLMNPQRSTVWY") # standard 20 amino acids

def fasta_iter(path: str) -> Iterator[Tuple[str, str]]:
    """Yield (header, seq) from (gz) FASTA."""
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        head, seq = None, []
        for line in f:
            line = line.rstrip("\n")
            if not line: continue
            if line.startswith(">"):
                if head: yield head, "".join(seq)
                head, seq = line[1:], []
            else:
                seq.append(line)
        if head: yield head, "".join(seq)

def parse_uniprot_id(header: str) -> str:
    # sp|P12345|ENTRY  or  tr|A0A…|ENTRY
    m = re.match(r"^(sp|tr)\|([^|]+)\|", header)
    return m.group(2) if m else header.split()[0]

def clean_seq(seq: str) -> str:
    s = seq.strip().upper()
    return "".join(c if c in AMINO else "X" for c in s)

def mean_pool_last_hidden(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    # [B,T,H], [B,T] -> [B,H]
    mask = attn_mask.unsqueeze(-1)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts

def chunk_sequence(seq: str, max_len: int, overlap: int = 0) -> List[str]:
    """Split a long sequence into chunks <= max_len (reserving BOS/EOS separately)."""
    if len(seq) <= max_len: return [seq]
    step = max_len - overlap
    return [seq[i:i+max_len] for i in range(0, len(seq), step)]

def _forward(model, tokenizer, seqs: List[str], device: torch.device, pos_limit: int):
    # IMPORTANT: use the model’s positional limit, not tokenizer.model_max_length
    toks = tokenizer(
        seqs, return_tensors="pt", padding=True, truncation=True, max_length=pos_limit
    )
    toks = {k: v.to(device) for k, v in toks.items()}
    with torch.no_grad():
        out = model(**toks, output_hidden_states=False)
    return mean_pool_last_hidden(out.last_hidden_state, toks["attention_mask"])

def embed_one(model, tokenizer, seq: str, device: torch.device, pos_limit: int) -> torch.Tensor:
    """Handles chunking if sequence is longer than model positional limit."""
    # Reserve 2 tokens for BOS/EOS (ESM uses special tokens)
    body_max = max(1, pos_limit - 2)
    if len(seq) <= body_max:
        return _forward(model, tokenizer, [seq], device, pos_limit)[0]
    parts = chunk_sequence(seq, body_max, overlap=0)
    outs = [_forward(model, tokenizer, [p], device, pos_limit)[0] for p in parts]
    return torch.stack(outs, dim=0).mean(dim=0)

def embed_batch(model, tokenizer, ids: List[str], seqs: List[str],
                out_dtype: torch.dtype, device: torch.device, pos_limit: int) -> Dict[str, np.ndarray]:
    out = {}
    for uid, s in zip(ids, seqs):
        v = embed_one(model, tokenizer, s, device, pos_limit).to(out_dtype).cpu().numpy()
        out[uid] = v
    return out

def ensure_out_root(raw_path: str) -> Path:
    p = Path(raw_path or "./data/embeddings").expanduser().resolve()
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except PermissionError:
        fb = Path("./embeddings").resolve()
        fb.mkdir(parents=True, exist_ok=True)
        print(f"[warn] Permission denied for '{p}'. Falling back to '{fb}'.")
        return fb

def pick_device_and_dtype(gpu_index: int):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_index}"), [torch.bfloat16, torch.float16, torch.float32]
    return torch.device("cpu"), [torch.float32]

def _from_pretrained(Cls, model_dir, dt, trust=False):
    # handle older transformers that don't accept `dtype` arg
    try:
        return Cls.from_pretrained(model_dir, local_files_only=False, dtype=dt, trust_remote_code=trust)
    except TypeError:
        try:
            return Cls.from_pretrained(model_dir, local_files_only=False, torch_dtype=dt, trust_remote_code=trust)
        except TypeError:
            m = Cls.from_pretrained(model_dir, local_files_only=False, trust_remote_code=trust)
            return m.to(dt)

def load_model_local(model_dir: str, device, dtype_candidates):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=False)
    last_err = None

    for dt in dtype_candidates:
        for trust in (False, True):
            if EsmModel is not None:
                try:
                    model = _from_pretrained(EsmModel, model_dir, dt, trust=trust).to(device).eval()
                    return model, tokenizer, dt
                except Exception as e:
                    last_err = e
            try:
                model = _from_pretrained(AutoModel, model_dir, dt, trust=trust).to(device).eval()
                return model, tokenizer, dt
            except Exception as e:
                last_err = e
            if EsmForMaskedLM is not None:
                try:
                    model = _from_pretrained(EsmForMaskedLM, model_dir, dt, trust=trust).to(device).eval()
                    return model, tokenizer, dt
                except Exception as e:
                    last_err = e
            try:
                model = _from_pretrained(AutoModelForMaskedLM, model_dir, dt, trust=trust).to(device).eval()
                return model, tokenizer, dt
            except Exception as e:
                last_err = e
    raise RuntimeError(f"Failed to load model from '{model_dir}' on {device}. Last error: {last_err}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="LOCAL path to ESM2 (e.g., /.../esm_models/15B)")
    ap.add_argument("--fasta", nargs="+", required=True, help="FASTA(.gz) files (Swiss-Prot/TrEMBL)")
    ap.add_argument("--out_root", default="./data/embeddings", help="Output root dir")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)         # kept for API compat; per-seq chunking inside
    ap.add_argument("--shard_size", type=int, default=200_000)    # #IDs per pickle
    args = ap.parse_args()

    out_root = ensure_out_root(args.out_root)
    device, dtype_candidates = pick_device_and_dtype(args.gpu)
    model, tokenizer, used_dtype = load_model_local(args.model_dir, device, dtype_candidates)

    # determine model positional limit
    pos_limit = int(getattr(model.config, "max_position_embeddings", 1024))

    print(f"[info] device={device}  dtype={used_dtype}  hidden={model.config.hidden_size}  pos_limit={pos_limit}")

    manifest = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "dtype": str(used_dtype).replace("torch.", ""),
        "device": str(device),
        "hidden_size": int(model.config.hidden_size),
        "pos_limit": pos_limit,
        "shards": [],
        "total": 0,
    }

    shard: Dict[str, np.ndarray] = {}
    shard_idx, total = 1, 0

    def flush():
        nonlocal shard, shard_idx
        if not shard: return
        out_p = out_root / f"protein_vec__{Path(args.model_dir).name}__shard_{shard_idx:05d}.pkl"
        with open(out_p, "wb") as f:
            pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)
        manifest["shards"].append({"path": str(out_p), "count": len(shard)})
        shard, shard_idx = {}, shard_idx + 1

    for fp in args.fasta:
        print(f"[stream] {fp}")
        for header, raw_seq in tqdm(fasta_iter(fp), unit="seq"):
            uid = parse_uniprot_id(header)
            if not uid: continue
            seq = clean_seq(raw_seq)
            try:
                vec_map = embed_batch(
                    model, tokenizer, [uid], [seq],
                    out_dtype=torch.float32, device=device, pos_limit=pos_limit
                )
            except Exception as e:
                print(f"[warn] failed {uid}: {e}")
                continue
            shard.update(vec_map)
            total += 1
            if len(shard) >= args.shard_size:
                flush()

    flush()
    manifest["total"] = total
    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] total={total}  shards={len(manifest['shards'])}  out_root={out_root}")

if __name__ == "__main__":
    # helps matmul stability/speed on Ampere+
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()


'''
conda activate chem
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python embed.py \
  --model_dir /home/danny/PROJECTS/MPI-VGAE/esm_models/15B \
  --fasta /data/uniprot/uniprot_sprot.fasta.gz /data/uniprot/uniprot_trembl.fasta.gz \
  --out_root /data/esm2_embeddings/t48_15B \
  --gpu 0 --batch_size 8 --shard_size 200000
'''

'''
THIS WORKED FOR ME, this is someone else's code i have reused
 /usr/bin/python3 embed.py   --model_dir facebook/esm2_t33_650M_UR50D   --fasta protein_names.fasta --out_root .
 the facebook/esm2_t33_650M_UR50D  model is downloaded from huggingface automatically. You can change it to point to your local esm2 model path if you have it downloaded already.
'''