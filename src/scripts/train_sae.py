"""Train an SAE on embeddings generated from r/chatGPT subreddit

Uses embeddings generated from model mlx-community/Qwen3-Embedding-8B-4bit-DWQ
SAE implementation from https://github.com/rmovva/HypotheSAEs
"""

import os
from pathlib import Path
import pickle
import numpy as np
from typing import List
from numpy.lib.npyio import NpzFile
from sklearn.model_selection import train_test_split
import json
from scripts.sae_utils import compute_feature_stats, load_split, load_trained_sae, select_top_features
from third_party.interpret_neurons import NeuronInterpreter, InterpretConfig, SamplingConfig, LLMConfig
import argparse
from third_party.sae import get_sae_checkpoint_name, load_model, train_sae

# M, K = 128, 4
TASK_SPECIFIC_INSTRUCTIONS = """All of the texts are posts related to sycophancy from the r/chatGPT Reddit subthread.
Features should describe a specific aspect of the post. For example:
- "mentions feeling alone"
- "mentions being frustrated with the new model update'\""""


def main():
    parser = argparse.ArgumentParser(description="Train an SAE on embeddings generated from r/chatGPT subreddit")
    parser.add_argument("-e", "--embed_path", type=str, default="data/embeddings_sycophant_posts.npz",
        help="Path to the embeddings NPZ file")
    parser.add_argument("-n", "--neuron_size", type=int, default=128,
        help="Number of neurons in the SAE")
    parser.add_argument("-k", "--top-k-neurons", type=int, default=4,
        help="Number of top neurons to keep during training loop")
    parser.add_argument("-m", "--max-epochs", type=int, default=100,
        help="Maximum number of epochs to train for")
    parser.add_argument("-c", "--checkpoint-dir", type=str, default=".checkpoints/reddit_comments_embed",
        help="Directory to save checkpoints")
    parser.add_argument("-nt", "--no-train", action="store_true", default=False,
        help="Whether to load a checkpoint instead of training a new model")
    args = parser.parse_args()
    
    train_data, val_data, train_embeddings, val_embeddings = load_split(args.embed_path)
    print(f"train_embed shape: {train_embeddings.shape}")
    print(f"val_embeddings shape: {val_embeddings.shape}")

    # prefix_lengths = [32, 128]
    prefix_lengths = None
    CACHE_NAME = "reddit_posts_embed"
    # CACHE_NAME = "reddit_posts_embed"
    # checkpoint_dir = f'.checkpoints/{CACHE_NAME}'
    if args.no_train:
        model = load_trained_sae(args.neuron_size, args.top_k_neurons, args.checkpoint_dir)
    else:
        model = train_sae(
            embeddings=train_embeddings,
            M=args.neuron_size,
            K=args.top_k_neurons,
            matryoshka_prefix_lengths=prefix_lengths,
            checkpoint_dir=args.checkpoint_dir,
            val_embeddings=val_embeddings,
            n_epochs=args.max_epochs

        )

    top_n_report = 20
    _, feature_means, feature_n_active = compute_feature_stats(model, train_embeddings)
    top_idx = select_top_features(feature_means, top_n_report)
    print(f"\nTop {len(top_idx)} features by mean activation:")
    for idx in top_idx:
        print(f"  neuron {idx:4d} | mean_activation={feature_means[idx]:.4f} | n_active={int(feature_n_active[idx])}")
 
    print(f"\nCheckpoint available under {args.checkpoint_dir} (M={args.neuron_size}, K={args.top_k_neurons}).")
    print("Run interpret_sae_features.py with the same -e / -n / -k / -c to interpret these features via LLM.")

if __name__ == '__main__':
    main()