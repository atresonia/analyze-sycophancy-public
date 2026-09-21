"""Assemble the candidate corpus: lexicon hits (Route L) plus direct replies to those hits (Route T).

A comment is a reply candidate if its parent_id points at a lexicon-hit comment (t1_<id>) or a lexicon-hit
post (t3_<id>) and it is not a hit itself. Replies are capped per thread (link_id), keeping the highest-scoring;
lexicon hits themselves are never capped.

Every row keeps enough provenance to rebuild subsets after one embedding run, e.g.
    v1 only:       route == 'lexicon' & in_v1
    v2 only:       route == 'lexicon'
    v2 + replies:  all rows
    v1 + replies:  (route == 'lexicon' & in_v1) | (route == 'reply' & parent_in_v1)

Outputs (in --out_dir):
    candidates.parquet          id, is_post, route, tier, terms, in_v1, parent_tier, parent_in_v1, link_id, parent_id, created_utc, score
    candidates_for_embed.jsonl  id, text, is_post, token_length  (upload this to Colab)
    capped_threads.csv          threads over the cap: link_id, permalink, n_replies, n_kept, n_dropped
    capped_dropped.parquet      every reply dropped by the cap: id, link_id, parent_id, score

Usage:
    python src/data_preprocessing/build_candidates.py            # cap 200
    python src/data_preprocessing/build_candidates.py --cap 100
"""

import argparse
import json
import os
import time
import pandas as pd

TEXT_FILES = ["data/text_comments_for_embed_2022-12-22_2026-9-18.jsonl",
              "data/text_posts_for_embed_2022-12-22_2026-9-18.jsonl"]
META_COLS = ["id", "link_id", "parent_id", "created_utc", "score"]


def find_replies(hits: pd.DataFrame, meta_comments: pd.DataFrame) -> pd.DataFrame:
    """Comments whose parent is a lexicon hit and that are not hits themselves, tagged with the parent's tier/in_v1."""
    parents = hits.assign(parent_id=hits.is_post.map({True: "t3_", False: "t1_"}) + hits.id)
    parents = parents[["parent_id", "tier", "in_v1"]].rename(columns={"tier": "parent_tier", "in_v1": "parent_in_v1"})
    replies = meta_comments[META_COLS][meta_comments.parent_id.isin(parents.parent_id) & ~meta_comments.id.isin(hits.id)]
    replies = replies.merge(parents, on="parent_id")
    return replies.assign(is_post=False, route="reply", tier=0, in_v1=False, terms=[[] for _ in range(len(replies))])


def cap_threads(replies: pd.DataFrame, cap: int, meta_posts: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Keep at most `cap` replies per link_id (highest score first) and write reports of what was dropped."""
    ranked = replies.sort_values(["link_id", "score"], ascending=[True, False])
    rank = ranked.groupby("link_id").cumcount()
    kept, dropped = ranked[rank < cap], ranked[rank >= cap]
    per_thread = ranked.groupby("link_id").size().rename("n_replies").loc[lambda s: s > cap].to_frame()
    if len(per_thread):
        per_thread["n_kept"] = cap
        per_thread["n_dropped"] = per_thread.n_replies - cap
        per_thread.insert(0, "permalink", meta_posts.set_index("t3_" + meta_posts.id).permalink.reindex(per_thread.index))
        per_thread.sort_values("n_dropped", ascending=False).to_csv(os.path.join(out_dir, "capped_threads.csv"))
        dropped[["id", "link_id", "parent_id", "score"]].to_parquet(os.path.join(out_dir, "capped_dropped.parquet"), index=False)
    print(f"cap={cap}: {len(dropped)} replies dropped from {len(per_thread)} threads (capped_threads.csv / capped_dropped.parquet)")
    return kept


def export_texts(candidate_ids: set, out_path: str) -> set:
    """Copy the embedding-ready rows of the candidate ids to out_path; returns the ids actually found."""
    found = set()
    with open(out_path, "w", encoding="utf-8") as out:
        for path in TEXT_FILES:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    row_id = json.loads(line)["id"]
                    if row_id in candidate_ids:
                        found.add(row_id)
                        out.write(line if line.endswith("\n") else line + "\n")
    return found


def main():
    parser = argparse.ArgumentParser(description="Assemble lexicon hits + direct replies into a candidate corpus")
    parser.add_argument("--hits", default="data/candidates/lexicon_hits.parquet")
    parser.add_argument("--meta_comments", default="data/meta_comments.parquet")
    parser.add_argument("--meta_posts", default="data/meta_posts.parquet")
    parser.add_argument("--out_dir", default="data/candidates")
    parser.add_argument("--cap", type=int, default=200, help="Max replies kept per thread (link_id)")
    args = parser.parse_args()
    t0 = time.time()

    hits = pd.read_parquet(args.hits).assign(route="lexicon", parent_tier=0, parent_in_v1=False)
    meta_comments = pd.read_parquet(args.meta_comments).drop_duplicates("id")  # raw dump repeats ~5k records
    meta_posts = pd.read_parquet(args.meta_posts).drop_duplicates("id")

    replies = find_replies(hits, meta_comments)
    print(f"lexicon hits: {len(hits)} | direct replies: {len(replies)} "
          f"(to hit comments: {replies.parent_id.str.startswith('t1_').sum()}, to hit posts: {replies.parent_id.str.startswith('t3_').sum()})")
    replies = cap_threads(replies, args.cap, meta_posts, args.out_dir)

    # attach thread/score metadata to the lexicon hits; comments and posts separately because their ids
    # share one namespace (a post is its own thread)
    post_meta = meta_posts[["id", "created_utc", "score"]].assign(link_id="t3_" + meta_posts.id, parent_id=None)
    hits = pd.concat([hits[~hits.is_post].merge(meta_comments[META_COLS], on="id", how="left"),
                      hits[hits.is_post].merge(post_meta, on="id", how="left")])
    cols = ["id", "is_post", "route", "tier", "terms", "in_v1", "parent_tier", "parent_in_v1", "link_id", "parent_id", "created_utc", "score"]
    candidates = pd.concat([hits[cols], replies[cols]], ignore_index=True)

    found = export_texts(set(candidates.id), os.path.join(args.out_dir, "candidates_for_embed.jsonl"))
    n_missing = (~candidates.id.isin(found)).sum()
    candidates = candidates[candidates.id.isin(found)]
    candidates.to_parquet(os.path.join(args.out_dir, "candidates.parquet"), index=False)

    print(f"{n_missing} replies had no row in the text files (deleted/short/bot, dropped upstream) -> excluded")
    print(f"\ncandidates: {len(candidates)} docs, {candidates.link_id.nunique()} threads, in {time.time() - t0:.0f}s")
    print(candidates.groupby(["route", "is_post"]).size().to_string())
    print(f"\nreplies by parent tier:\n{candidates[candidates.route == 'reply'].parent_tier.value_counts().sort_index().to_string()}")
    print(f"\nsubset sizes:  v1={((candidates.route == 'lexicon') & candidates.in_v1).sum()}  v2={(candidates.route == 'lexicon').sum()}  "
          f"v1+T={(((candidates.route == 'lexicon') & candidates.in_v1) | ((candidates.route == 'reply') & candidates.parent_in_v1)).sum()}  "
          f"v2+T={len(candidates)}")


if __name__ == "__main__":
    main()
