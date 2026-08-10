"""Interpret neurons specified for SAE. Requires checkpoint for model (no training)"""
import os
import json
import argparse
 
from sae_utils import (
    load_split,
    load_trained_sae,
    compute_feature_stats,
    select_top_features,
    load_texts_for_ids,
)
from third_party.interpret_neurons import NeuronInterpreter, InterpretConfig, SamplingConfig, LLMConfig

# M, K = 128, 4
TASK_SPECIFIC_INSTRUCTIONS = """All of the texts are posts related to sycophancy from the r/chatGPT Reddit subthread.
Features should describe a specific aspect of the post. For example:
- "mentions feeling alone"
- "mentions being frustrated with the new model update'\""""

def main():
    parser = argparse.ArgumentParser(description="Interpret trained SAE features via LLM")
    parser.add_argument("-e", "--embed_path", type=str, required=True, help="Path to the embeddings NPZ file")
    parser.add_argument("-i", "--text_file", type=str, required=True,
                         help="Source JSONL the embeddings were generated from (for id -> text lookup)")
    parser.add_argument("-n", "--neuron_size", type=int, default=128)
    parser.add_argument("-k", "--top-k-neurons", type=int, default=4)
    parser.add_argument("-c", "--checkpoint-dir", type=str, default=".checkpoints/reddit_comments_embed")
    parser.add_argument("-t", "--top-n-features", type=int, default=20, help="How many top features to interpret")
    parser.add_argument("--interpreter-model", type=str, default="gpt-5.2")
    parser.add_argument("--annotator-model", type=str, default="gpt-5-mini")
    parser.add_argument("--n-candidates", type=int, default=3, help="Candidate interpretations per neuron")
    parser.add_argument("--cache-name", type=str, default="reddit_comments_embed",
                         help="Cache key for NeuronInterpreter's on-disk annotation cache")
    parser.add_argument("-o", "--output_file", type=str, default=None,
                         help="Where to save interpretations JSON (default: <cache-name>_interpretations.json)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_KEY_SAE") and not os.environ.get("OPENAI_BASE_URL"):
        raise SystemExit(
            "OPENAI_KEY_SAE is not set (and no OPENAI_BASE_URL local endpoint configured). "
            "See https://github.com/rmovva/HypotheSAEs#setup."
        )
    
    train_data, _, train_embeddings, _ = load_split(args.embed_path)
    model = load_trained_sae(args.neuron_size, args.top_k_neurons, args.checkpoint_dir)
    activations, feature_means, _ = compute_feature_stats(model, train_embeddings)
    top_idx = select_top_features(feature_means, args.top_n_features)
    print(f"Interpreting {len(top_idx)} features: {top_idx}")

    train_texts = load_texts_for_ids(args.text_file, train_data["ids"])

    print(f"Loaded {len(train_texts)} aligned texts for interpretation")
 
    interpreter = NeuronInterpreter(
        interpreter_model=args.interpreter_model,
        annotator_model=args.annotator_model,
        n_workers_interpretation=10,
        n_workers_annotation=50,
        cache_name=args.cache_name,
    )
    interpret_config = InterpretConfig(
        sampling=SamplingConfig(n_examples=20, max_words_per_example=128),
        llm=LLMConfig(temperature=0.7, max_interpretation_tokens=75),
        n_candidates=args.n_candidates,
        task_specific_instructions=TASK_SPECIFIC_INSTRUCTIONS,
    )
 
    interpretations = interpreter.interpret_neurons(
        texts=train_texts,
        activations=activations,
        neuron_indices=top_idx,
        config=interpret_config,
    )
 
    output_file = args.output_file or f"{args.cache_name}_interpretations.json"
    with open(output_file, "w") as f:
        json.dump({str(k): v for k, v in interpretations.items()}, f, indent=2)
    print(f"Saved -> {output_file}")
 
    for idx, candidates in interpretations.items():
        print(f"  neuron {idx}: {candidates}")
 
 
if __name__ == "__main__":
    main()
 