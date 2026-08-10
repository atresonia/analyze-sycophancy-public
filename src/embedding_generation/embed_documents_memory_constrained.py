"""
Encodes text data into embeddings using the nemotron embedding model.

Note: we need to use ncorder/llama-embed-nemotron-8b-mlx-8bit instead of nvidia/llama-embed-nemotron-8b
because we do not have enough memory on our hardware

Usage:
 - python embed_documents_memory_constrained.py
 - python embed_documents_memory_constrained.py -i data/text_posts_for_embed.jsonl -o data/embeddings_posts.npz
 - python embed_documents_memory_constrained.py -i data/text_sycophant_comments_for_embed.jsonl -o data/embeddings_sycophant_comments.npz
 - python embed_documents_memory_constrained.py -i data/text_sycophant_combined_for_embed.jsonl -o data/embeddings_sycophant_combined.npz
"""

import json
from mlx_embeddings.utils import load
import numpy as np
import mlx.core as mx
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import psutil
import time
from tqdm import tqdm

MODEL_NAME = "mlx-community/Qwen3-Embedding-8B-4bit-DWQ"
BATCH_SIZE = 48
MAX_LENGTH = 2048
INSTRUCTION = "Instruct: Identify the main topic or concept expressed in the text\nQuery:"
CLEAR_CACHE_EVERY = 10


def embed_text(texts, model, tokenizer, batch_size, max_length, instruction=INSTRUCTION):
    # sort by length to reduce padding waste per batch
    order = sorted(range(len(texts)), key=lambda x: len(texts[x]))
    embeds = [None] * len(texts)
    n_batches = (len(order) + batch_size - 1) // batch_size
    starts = range(0, len(order), batch_size)
    starts = tqdm(starts, total=n_batches, unit="batch", desc="Embedding")
    t_loop = time.time()

    for bi, start in enumerate(starts):
        batch_idx = order[start: start + batch_size]
        batch_texts = [instruction + texts[i] for i in batch_idx]
        inputs = tokenizer.batch_encode_plus(
            batch_texts, 
            return_tensors="mlx",
            padding=True, 
            truncation=True, 
            max_length=max_length
        )
        outputs = model(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"] 
        )
        mx.eval(outputs.text_embeds)
        batch_embeds = np.array(outputs.text_embeds.astype(mx.float16), copy=False)
        for j, idx in enumerate(batch_idx):
            embeds[idx] = batch_embeds[j]
        if (bi + 1) % CLEAR_CACHE_EVERY == 0:
            try:
                mx.clear_cache()
            except AttributeError:
                pass
    return np.stack(embeds)


# TODO: this is very inefficient and crashed for text_posts_for_embed.jsonl, need to evaluate alternative approaches
def get_embedding(texts, model, tokenizer, batch_size: int = BATCH_SIZE, max_length: int = MAX_LENGTH):
    ordered_indices = sorted(range(len(texts)), key=lambda x: len(texts[x]))
    texts_sorted = [texts[i] for i in ordered_indices]
    all_embeds = [None] * len(texts)

    for i in range(0, len(texts_sorted), batch_size):
        batch = texts_sorted[i:i+batch_size]
        inputs = tokenizer.batch_encode_plus(
            batch, 
            return_tensors="mlx",
            padding=True, 
            truncation=True, 
            max_length=max_length
        )
        outputs = model(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"] 
        )
        mx.eval(outputs.text_embeds)

        batch_embeds = np.array(outputs.text_embeds.astype(mx.float32), copy=False)
        batch_indices = ordered_indices[i:i+batch_size]
        for i, idx in enumerate(batch_indices):
            all_embeds[idx] = batch_embeds[i]
    return np.stack(all_embeds)


def compute_and_print_similarity(embeddings):
    B, D = embeddings.shape
    similarity_matrix = cosine_similarity(embeddings)
    # print(f"Similarity matrix between {B} documents: {similarity_matrix}")

    # for i in range(B):
    #     for j in range(i+1, B):
    #         print(f"Similarity between sequence {i + 1} and sequence {j + 1}: {similarity_matrix[i][j]:.4f}")
    return similarity_matrix


def plot_similarity_matrix(similarity_matrix, labels):
    plt.figure(figsize=(5, 4))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', xticklabels=labels, yticklabels=labels)
    plt.title('Similarity Matrix Heatmap')
    plt.tight_layout()
    plt.show()


def load_texts(input_path: str, load_count: int = None):
    ids, texts, is_post, token_length = [], [], [], []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ids.append(row['id'])
            texts.append(row['text'])
            is_post.append(row['is_post'])
            token_length.append(row['token_length'])
            if load_count is not None and len(ids) >= load_count:
                break
    return ids, texts, is_post, token_length

def main():
    start_time = time.time()
    print(f"Starting time: {start_time}")
    parser = argparse.ArgumentParser(description="Given a JSONL file of texts, generate embeddings and save them to a NPZ file")
    parser.add_argument("-i", "--input_file", type=str, default="data/text_sycophan_posts_for_embed.jsonl", 
    help="Input JSONL file of texts")
    parser.add_argument("-n", "--load_input_size", type=int, default=None)
    parser.add_argument("-o", "--output_file", type=str, default="data/embeddings/sycophan_posts_2026-8-7.npz")
    parser.add_argument("-m", "--model_name", type=str, default="mlx-community/Qwen3-Embedding-8B-4bit-DWQ",
    help="Name of the model to use for embeddings generation")
    parser.add_argument("-b", "--batch_size", type=int, default=BATCH_SIZE,
    help="Batch size for embeddings generation")
    parser.add_argument("-l", "--max_length", type=int, default=MAX_LENGTH,
    help="Maximum length of the text to be embedded")
    parser.add_argument("--no_instruction", action="store_true", help="Embed raw text with no instruction prefix")
    args = parser.parse_args()
    instruction = "" if args.no_instruction else INSTRUCTION
    print(f"Loading model {args.model_name} ...")
    model, tokenizer = load(args.model_name)
    print(f"  RSS after model load: {psutil.Process().memory_info().rss / 1024**3:.2f} GB ")

    ids, texts, is_post, token_length = load_texts(args.input_file, args.load_input_size)
    print(f"Loaded {len(texts)} texts from {args.input_file}")

    t0 = time.time()
    # embeddings = get_embedding(docs, model, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    embeddings = embed_text(texts, model, tokenizer, batch_size=args.batch_size, 
                            max_length=args.max_length, instruction=instruction)
    print(f"Embedded {embeddings.shape[0]} docs (dim={embeddings.shape[1]}) in {time.time() - t0:.1f}s")
    assert embeddings.shape[0] == len(texts), "Number of embeddings and ids do not match"
    # similarity_matrix = compute_and_print_similarity(embeddings)
    # labels = [f"Doc {i+1}" for i in range(len(docs))]
    # plot_similarity_matrix(similarity_matrix, labels)

    np.savez(
        args.output_file,
        ids=np.array(ids, dtype=object),
        embeddings=embeddings,
        is_post=np.array(is_post, dtype=bool),
        token_length=np.array(token_length, dtype=int),
    )
    print(f"Saved -> {args.output_file}")

    print(f"Final RSS: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")


if __name__ == '__main__':
    main()