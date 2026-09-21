"""
Encodes text data into embeddings with a GPU (written for Google Colab, A100).

Uses the official bf16 nvidia/llama-embed-nemotron-8b via sentence-transformers (the MLX 8-bit port in
embed_documents_memory_constrained.py is for the laptop only). Documents are embedded without an
instruction prefix, as the model card specifies for documents.

Texts are sorted by token_length and written in shards, so a Colab disconnect only loses the current
shard: re-running the same command skips shards that already exist. The final .npz holds {ids, embeddings}.

Colab usage:
    !pip install -q -U sentence-transformers
    from huggingface_hub import login; login()          # model is gated: accept the licence on HF first
    from google.colab import drive; drive.mount('/content/drive')
    !python embed_documents.py -i /content/drive/MyDrive/sycophancy/candidates_for_embed.jsonl \
                               -o /content/drive/MyDrive/sycophancy/embeddings_candidates_nemotron.npz
"""

import argparse
import json
import os
import time
import numpy as np
import torch
from packaging.version import Version
from sentence_transformers import SentenceTransformer, __version__ as st_version

MODEL_NAME = "nvidia/llama-embed-nemotron-8b"
BATCH_SIZE = 32
MAX_LENGTH = 2048
SHARD_SIZE = 2000


def load_texts(input_path: str):
    ids, texts, token_length = [], [], []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ids.append(row["id"])
            texts.append(row["text"])
            token_length.append(row.get("token_length", len(row["text"])))
    return ids, texts, token_length


def main():
    parser = argparse.ArgumentParser(description="Embed a JSONL of texts on a GPU, sharded and resumable")
    parser.add_argument("-i", "--input_file", default="data/candidates/candidates_for_embed.jsonl")
    parser.add_argument("-o", "--output_file", default="data/embeddings/embeddings_candidates_nemotron.npz")
    parser.add_argument("-m", "--model_name", default=MODEL_NAME)
    parser.add_argument("-b", "--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("-l", "--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("-s", "--shard_size", type=int, default=SHARD_SIZE)
    parser.add_argument("-n", "--load_input_size", type=int, default=None, help="Only embed the first n texts (smoke test)")
    args = parser.parse_args()

    ids, texts, token_length = load_texts(args.input_file)
    if args.load_input_size:
        ids, texts, token_length = ids[:args.load_input_size], texts[:args.load_input_size], token_length[:args.load_input_size]
    # longest first: shards are length-homogeneous (little padding) and an OOM shows up immediately
    order = sorted(range(len(texts)), key=lambda i: -token_length[i])
    print(f"Loaded {len(texts)} texts ({sum(token_length)} tokens) from {args.input_file}")

    shard_dir = os.path.splitext(args.output_file)[0] + "_shards"
    os.makedirs(shard_dir, exist_ok=True)
    shards = [order[s:s + args.shard_size] for s in range(0, len(order), args.shard_size)]
    todo = [k for k in range(len(shards)) if not os.path.exists(os.path.join(shard_dir, f"shard_{k:04d}.npy"))]
    print(f"{len(shards)} shards of <= {args.shard_size}; {len(shards) - len(todo)} already done")

    if todo:
        # padding_side left as in the model card; kwarg was renamed in sentence-transformers 5.6
        pad_kw = "processor_kwargs" if Version(st_version) >= Version("5.6") else "tokenizer_kwargs"
        model = SentenceTransformer(
            args.model_name, trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"},
            **{pad_kw: {"padding_side": "left"}},
        )
        model.max_seq_length = min(args.max_length, model.get_max_seq_length() or args.max_length)
        encode = getattr(model, "encode_document", model.encode)
        t0, done_tokens = time.time(), 0
        for k in todo:
            idx = shards[k]
            emb = encode([texts[i] for i in idx], batch_size=args.batch_size,
                         convert_to_numpy=True, normalize_embeddings=True)
            np.save(os.path.join(shard_dir, f"shard_{k:04d}.npy"), emb.astype(np.float16))
            done_tokens += sum(token_length[i] for i in idx)
            print(f"shard {k + 1}/{len(shards)} done | {done_tokens / (time.time() - t0):.0f} tok/s | "
                  f"{time.time() - t0:.0f}s elapsed", flush=True)

    # stitch shards back into the input order
    embeddings = np.concatenate([np.load(os.path.join(shard_dir, f"shard_{k:04d}.npy")) for k in range(len(shards))])
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order))
    embeddings = embeddings[inverse]
    assert embeddings.shape[0] == len(ids), "Number of embeddings and ids do not match"

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    np.savez(args.output_file, ids=np.array(ids, dtype=object), embeddings=embeddings)
    print(f"Saved {embeddings.shape} -> {args.output_file}")


if __name__ == "__main__":
    main()
