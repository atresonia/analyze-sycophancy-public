# CLAUDE.md

Research code for "Public View of Sycophancy in LLMs": select sycophancy-related r/ChatGPT posts/comments with a
tiered lexicon, embed them (nemotron-8b on Colab), train a sparse autoencoder on the embeddings, interpret the features.
`README.md` documents the pipeline step by step with the current counts; `docs/lexicon.md` is the lexicon record.

## Setup and commands
- `pip install -e .` once; run every script from the repo root. Packages: `data_preprocessing`, `embedding_generation`, `scripts`, `third_party` (all under `src/`).
- Corpus pipeline, in order (defaults point at `data/`):
  1. `python src/data_preprocessing/input_to_embeddings.py -i <raw dump(s)> -o <text jsonl>` (~1 h; do not re-run casually)
  2. `python src/data_preprocessing/build_metadata.py` (10 s)
  3. `python src/data_preprocessing/build_lexicon.py` (log-odds term ranking, ~5 min; output is for hand selection only)
  4. `python src/data_preprocessing/filter_lexicon.py` (~11 min)
  5. `python src/data_preprocessing/build_candidates.py` (20 s)
- Embeddings: `src/embedding_generation/embed_documents.py` on Colab (A100). `embed_documents_memory_constrained.py` is the superseded laptop/MLX path.
- SAE: `python src/scripts/train_sae.py -e <npz> -n <M> -k <K> -c <checkpoint_dir>`; `interpret_sae.py` needs `OPENAI_API_KEY`.

## Layout
- `src/data_preprocessing/` corpus construction; `lexicon.py` is the single source of truth for the lexicon
- `src/embedding_generation/` embedding scripts
- `src/scripts/` SAE train / interpret; `src/third_party/` adapted from HypotheSAEs - keep its style, do not refactor
- `data/` gitignored. Raw dumps `r_chatGPT_*_2022-12-22_2026-9-18.jsonl`; text files `text_*_for_embed_*.jsonl`; sidecars `meta_*.parquet`; corpus in `data/candidates/`; embeddings in `data/embeddings/`
- `docs/notes/` earlier-stage write-ups (historical; do not update)

## Data notes
- Comment and post ids share one base-36 namespace: never join comments and posts on `id` in one table; merge per type.
- The raw comments dump repeats ~5k records: `drop_duplicates("id")` on `meta_comments.parquet` before any join.
- `text_*_for_embed_*.jsonl` carries only `{id, text, is_post, token_length}`; everything else is joined from the sidecars by `id`.
- Older text files (`text_comments_sycophan_for_embed.jsonl`) predate the bot filter and still contain AutoModerator rows.
- Deleted/removed parents are rare in the Arctic Shift dump (3 of ~8k checked): recover context from the dump, do not drop comments for missing parents.
- The `.npz` embeddings hold `ids` + `embeddings` only (fp16); `ids` order equals the input JSONL order.

## Conventions
- Scripts: short, a usage docstring with example commands, argparse with `data/` defaults, print counts of what was kept/dropped.
- Provenance columns (`route`, `tier`, `in_v1`, `parent_tier`, `parent_in_v1`) are labels for selecting subsets later, never filters inside the pipeline.
- Lexicon changes: keep the exact V1 patterns unchanged (they define the v1 corpus); add new terms to the `*_V2` lists; re-run steps 4-5 and confirm the subset sizes in `README.md` (v1 30,249 / v2 61,509 / v1+T 58,559 / v2+T 119,127) before accepting a change that should not alter them.
- Any change to the corpus must report per-tier / per-route counts and a random sample of what was added before it is accepted.
- Numbers in docs come from a printed run, not from arithmetic or memory.

## Working style
- Check in at the end of each pipeline stage and ask before moving to the next; the user approves edits individually.
- Push back with measured numbers (counts, timings, sampled precision), not opinions.
- Do not delete or regenerate files under `data/` without asking; do not add project documents to the repo unless asked.
- Preserve the user's wording in `README.md` and docs; edit only for factual errors, broken paths/commands, or missing information.
