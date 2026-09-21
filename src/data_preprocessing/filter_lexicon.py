"""Route L: select posts/comments whose text matches the sycophancy lexicon (see lexicon.py).

Reads the embedding-ready text files and writes, under --out_dir:
    lexicon_hits.parquet        id, is_post, token_length, tier (1 = jargon, 2 = phrase only, 3 = adjacent only),
                                terms (matched regex fragments), in_v1 (would have matched the original hand lexicon)
    lexicon_hits_for_embed.jsonl  the matching rows of the input files, unchanged (upload this to Colab)
and prints counts per tier and per term so noisy terms are easy to spot. Thread/score metadata is joined
later by id from data/meta_*.parquet (see build_metadata.py).

Usage:
    python src/data_preprocessing/filter_lexicon.py
    python src/data_preprocessing/filter_lexicon.py -i data/text_comments_for_embed.jsonl data/text_posts_for_embed.jsonl -o data/candidates
"""

import argparse
import collections
import json
import os
import time
import pandas as pd

from data_preprocessing.lexicon import ANY_RE, match


def scan_file(path: str, out) -> list[dict]:
    """Return lexicon hits for one JSONL text file and copy the matching lines to `out`. The regex is
    first run on the raw line so that json.loads only happens for the (rare) candidate lines."""
    hits = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not ANY_RE.search(line):
                continue
            row = json.loads(line)
            tier, terms, in_v1 = match(row["text"])
            if tier:
                hits.append(dict(id=row["id"], is_post=row["is_post"], token_length=row["token_length"],
                                 tier=tier, terms=terms, in_v1=in_v1))
                out.write(line if line.endswith("\n") else line + "\n")
    return hits


def report(df: pd.DataFrame) -> None:
    for is_post, label in [(False, "comments"), (True, "posts")]:
        sub = df[df.is_post == is_post]
        print(f"{label}: {len(sub)} hits  (tier 1 jargon: {(sub.tier == 1).sum()}, tier 2 phrase only: {(sub.tier == 2).sum()}, "
              f"tier 3 adjacent only: {(sub.tier == 3).sum()}; v1 lexicon alone: {sub.in_v1.sum()})")
    term_counts = collections.Counter(t for terms in df.terms for t in terms)
    print("\nhits per term (a doc can count for several):")
    for term, n in term_counts.most_common():
        print(f"  {n:7d}  {term}")


def main():
    parser = argparse.ArgumentParser(description="Select posts/comments matching the sycophancy lexicon")
    parser.add_argument("-i", "--input_files", nargs="+",
        default=["data/text_comments_for_embed_2022-12-22_2026-9-18.jsonl",
                 "data/text_posts_for_embed_2022-12-22_2026-9-18.jsonl"])
    parser.add_argument("-o", "--out_dir", default="data/candidates")
    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_path = os.path.join(args.out_dir, "lexicon_hits_for_embed.jsonl")
    parquet_path = os.path.join(args.out_dir, "lexicon_hits.parquet")
    with open(jsonl_path, "w", encoding="utf-8") as out:
        hits = [h for path in args.input_files for h in scan_file(path, out)]
    df = pd.DataFrame(hits)
    df.to_parquet(parquet_path, index=False)
    print(f"Scanned {len(args.input_files)} files in {time.time() - t0:.0f}s; "
          f"wrote {len(df)} hits -> {parquet_path} and {jsonl_path}\n")
    report(df)


if __name__ == "__main__":
    main()
