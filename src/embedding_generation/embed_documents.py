"""
Encodes text data into embeddings using the nemotron embedding model.
"""

import json
from sentence_transformers import SentenceTransformer
import torch
import os
import numpy as np

INPUT_PATH = 'data/texts.jsonl'
MODEL_NAME = 'nvidia/llama-embed-nemotron-8b'
EMBEDDINGS_OUT_PATH = 'data/embeddings.npz'

def load_texts(input_path: str):
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
    return ids, texts, is_post, token_length

def main():
    model = SentenceTransformer(
        MODEL_NAME, trust_remote_code=True,
        model_kwargs={"attn_implementation": "eager", "torch_dtype": torch.float32},
        tokenizer_kwargs={"padding_side": "left"}
    )
    # ids, texts, is_post, token_length = load_texts(INPUT_PATH)
    # grab a random entry from the input file and grab the 'text' field
    doc1 = json.load(open(INPUT_PATH, 'r', encoding='utf-8'))[0]['text']
    doc2 = json.load(open(INPUT_PATH, 'r', encoding='utf-8'))[1]['text']
    embeddings = model.encode_document(
        [doc1, doc2],
        batch_size=8,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"Embeddings shape: {embeddings.shape}")
    similarity_matrix = model.similarity(embeddings, embeddings)
    doc_similarity = similarity_matrix[0][1].item()
    print(f"Similarity between documents: {doc_similarity:.4f}")

    # np.savez(
    #     EMBEDDINGS_OUT_PATH,
    #     ids=np.array(ids, dtype=object),
    #     embeddings=embeddings,
    #     is_post=np.array(is_post, dtype=bool),
    #     token_length=np.array(token_length, dtype=int),
    # )
    # print(f"Saved embeddings to {EMBEDDINGS_OUT_PATH}")


if __name__ == '__main__':
    main()
