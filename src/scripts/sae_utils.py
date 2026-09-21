"""Shared functions for training SAE and interpretting neurons"""
import os
import json
from pathlib import Path

from numpy.lib.npyio import NpzFile
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Tuple

from third_party.sae import SparseAutoencoder, get_sae_checkpoint_name, load_model

VAL_RATIO = 0.2

def split_train_val_embeddings(embed_data: NpzFile, val_ratio=0.2) -> tuple[dict, dict]:
    """Take in data from .npz file (keys: ['ids', 'embeddings', 'is_post', 'token_length']) and split
    into train-val subsets
    
    Returns in format:
        - train_data: {'ids': [<train_ids>], 'embeddings': [<train_embeddings>], 'is_post': ..., 'token_length'}
        - val_data: {'ids': [<val_ids>], 'embeddings': [<val_embeddings>], 'is_post': ..., 'token_length'}
    """
    keys = embed_data.files
    arrays = [embed_data[key] for key in keys]
    n = len(arrays[0])
    idx_train, idx_val = train_test_split(np.arange(n), test_size=val_ratio, random_state=42)
    train_data = {k: arr[idx_train] for k, arr in zip(keys, arrays)}
    val_data = {k: arr[idx_val] for k, arr in zip(keys, arrays)}
    return train_data, val_data

def load_train_text(train_data: NpzFile, text_file: str) -> List[str]:
    """Get original text from train embeddings"""
    train_text = []
    train_embed_ids = train_data["ids"]
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data["id"] in train_embed_ids:
                train_text.append(data["text"])
    return train_text

def load_embeddings(emb_data: NpzFile) -> np.ndarray:
    """Return (n, d) float32 ndarray of embeddings.
    """
    embeddings = emb_data['embeddings']
    if embeddings.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {embeddings.shape}")
    return embeddings.astype(np.float32, copy=False)

def load_split(embed_path: str, val_ratio: float = VAL_RATIO):
    """Load embeddings .npz file and return (train_data, val_data, train_embeddings, val_embeddings).
    """
    embed_data = np.load(Path(embed_path), allow_pickle=True)
    train_data, val_data = split_train_val_embeddings(embed_data, val_ratio)
    return train_data, val_data, load_embeddings(train_data), load_embeddings(val_data)

def load_trained_sae(M, K, checkpoint_dir, prefix_lengths=None) -> SparseAutoencoder:
    """Load an already-trained checkpoint"""
    checkpoint_name = get_sae_checkpoint_name(M, K, prefix_lengths)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Run train_sae_run.py first with "
            f"-n {M} -k {K} -c {checkpoint_dir}."
        )
    return load_model(checkpoint_path)

def compute_feature_stats(model: SparseAutoencoder, train_embeddings) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (activations, feature_means, feature_c_active)"""
    activations = model.get_activations(train_embeddings) # (N, M): N=train sample size, M = sae dimension
    feature_means = activations.mean(axis=0) # (M,): calculates the mean activation for each neuron across all the samples
    feature_n_active = (activations > 0).sum(axis=0) # calculates the number of times each neuron i was active across the dataset: (M,)
    return activations, feature_means, feature_n_active

def select_top_features(feature_means: np.ndarray, n_select: int=20) -> List[int]:
    """Return top-N neurons by mean activation: most active neurons"""
    top_idx = np.argsort(-feature_means)[:n_select]
    return [int(idx) for idx in top_idx if feature_means[idx] > 0]

def load_texts_for_ids(text_file: str, ids) -> List[str]:
    """Load texts aligned to 'ids' so that the list matches the activations order"""
    id_to_text = {}
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            id_to_text[row["id"]] = row["text"]
    
    ids_list = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    missing = [i for i in ids_list if i not in id_to_text]
    if missing:
        raise KeyError(
            f"{len(missing)} ids from the embeddings file were not found in {text_file} "
            f"(e.g. {missing[:5]}). texts and activations must come from the same source data."
        )
    return [id_to_text[i] for i in ids_list]