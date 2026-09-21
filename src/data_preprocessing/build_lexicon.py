"""Find candidate lexicon terms by log-odds ratio (Monroe, Colaresi & Quinn 2008, informative Dirichlet prior).

Compares word/bigram frequencies in a positive set against a background sample:
    positive:   the 17k "sycophan" comments (keyword in the comment or in its post) - independent of the
                tier-2 phrases in lexicon.py, so those should surface here if they are real signal
    background: every k-th comment of the full 4M comment file (~200k)

Writes a CSV of the top terms (z-score, counts, whether lexicon.py already matches the term) for hand
selection; chosen terms are then added to lexicon.py by hand. Nothing here edits the lexicon.

Usage:
    python src/data_preprocessing/build_lexicon.py
    python src/data_preprocessing/build_lexicon.py --top 300 --min_pos_count 20
"""

import argparse
import collections
import json
import re
import numpy as np
import pandas as pd

from data_preprocessing.lexicon import ANY_RE

TOKEN_RE = re.compile(r"[a-z][a-z']+")
# common function words: skipped as unigrams and as either half of a bigram
STOP = set("""the a an and or but if of to in on at for with from by as is are was were be been being it its it's this that
these those i me my we our you your he she they them their his her im i'm ive i've dont don't do does did doesn't didn't not no
so just have has had can could would should will not than then there here what which who how when where why all any some very
more most much really also about into out up down over just like get got one thing things because too""".split())


def ngrams(text: str):
    toks = [t for t in TOKEN_RE.findall(text.lower())]
    uni = [t for t in toks if t not in STOP]
    bi = [f"{a} {b}" for a, b in zip(toks, toks[1:]) if a not in STOP and b not in STOP]
    return uni + bi


BOT_RE = re.compile(r"I am a bot, and this action was performed automatically|^Hey /u/", re.I)


def count_file(path: str, every: int = 1) -> tuple[collections.Counter, int]:
    """Count the frequency of each "text" field (unigram and bigram) in the file."""
    counts, n_docs = collections.Counter(), 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % every:
                continue
            text = json.loads(line)["text"]
            if BOT_RE.search(text):  # older text files predate the bot-author filter
                continue
            counts.update(set(ngrams(text)))  # document frequency, not token frequency
            n_docs += 1
    return counts, n_docs


def log_odds(pos: collections.Counter, bg: collections.Counter, alpha0: float = 500.0) -> pd.DataFrame:
    """z-scored log-odds of each term in pos vs bg, with a Dirichlet prior proportional to pooled frequencies."""
    terms = [t for t in pos if bg.get(t, 0) or pos[t]]
    y_i = np.array([pos[t] for t in terms], dtype=float)
    y_j = np.array([bg.get(t, 0) for t in terms], dtype=float)
    n_i, n_j = y_i.sum(), sum(bg.values())
    prior = (y_i + y_j) / (n_i + n_j) * alpha0
    a0 = prior.sum()
    delta = np.log((y_i + prior) / (n_i + a0 - y_i - prior)) - np.log((y_j + prior) / (n_j + a0 - y_j - prior))
    z = delta / np.sqrt(1 / (y_i + prior) + 1 / (y_j + prior))
    return pd.DataFrame(dict(term=terms, pos_docs=y_i.astype(int), bg_docs=y_j.astype(int), z=z))


def main():
    parser = argparse.ArgumentParser(description="Rank candidate lexicon terms by log-odds vs a background sample")
    parser.add_argument("--positive", default="data/text_comments_sycophan_for_embed.jsonl")
    parser.add_argument("--background", default="data/text_comments_for_embed_2022-12-22_2026-9-18.jsonl")
    parser.add_argument("--every", type=int, default=20, help="Background = every k-th comment")
    parser.add_argument("--min_pos_count", type=int, default=20)
    parser.add_argument("--top", type=int, default=300)
    parser.add_argument("-o", "--output_file", default="data/candidates/logodds_terms.csv")
    args = parser.parse_args()

    pos, n_pos = count_file(args.positive)
    bg, n_bg = count_file(args.background, args.every)
    print(f"positive: {n_pos} docs, {len(pos)} terms | background: {n_bg} docs, {len(bg)} terms")

    pos = collections.Counter({t: c for t, c in pos.items() if c >= args.min_pos_count})
    df = log_odds(pos, bg).sort_values("z", ascending=False).head(args.top)
    df["pos_rate"] = (df.pos_docs / n_pos).round(4)
    df["bg_rate"] = (df.bg_docs / n_bg).round(5)
    df["in_lexicon"] = df.term.map(lambda t: bool(ANY_RE.search(t)))
    df.to_csv(args.output_file, index=False)
    print(f"wrote top {len(df)} terms -> {args.output_file}\n")
    pd.set_option("display.width", 160)
    print(df.head(100).to_string(index=False))


if __name__ == "__main__":
    main()
