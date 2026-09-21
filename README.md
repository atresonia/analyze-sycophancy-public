# Train SAE on Sycophancy Dataset

How the r/ChatGPT public talks about LLM sycophancy. We select a sycophancy-related corpus from the full r/ChatGPT archive with a tiered lexicon (plus direct replies), embed it with `nvidia/llama-embed-nemotron-8b`, and train a sparse autoencoder (SAE) on the embeddings to discover the concepts in that discourse.

## Setup
- Python >= 3.11. From the repo root: `pip install -e .` (installs `src/` as the packages `data_preprocessing`, `embedding_generation`, `scripts`, `third_party`). Run all scripts from the repo root.
- `data/` is gitignored (raw dumps are 16 GB). The paths below are the defaults baked into each script.
- An OpenAI key (`OPENAI_API_KEY`) is only needed for `scripts/interpret_sae.py`.
- Embedding generation needs a GPU (Google Colab A100 was used); everything else runs on a laptop.

## Layout
- `src/data_preprocessing/` — corpus construction (steps 1-6 below)
- `src/embedding_generation/` — `embed_documents.py` (Colab, current); `embed_documents_memory_constrained.py` is the older laptop/MLX path
- `src/scripts/` — SAE training and interpretation
- `src/third_party/` — SAE implementation and neuron interpretation adapted from [HypotheSAEs](https://github.com/rmovva/HypotheSAEs)
- `docs/lexicon.md` — the lexicon terms with hit counts, spot-check precision and rejected terms; `docs/notes/` — earlier stage write-ups

## Data
Data was collected from the Arctic Shift archive of Reddit (a Pushshift-style dump, not the live Reddit API): https://arctic-shift.photon-reddit.com/download-tool. repo: https://github.com/ArthurHeitmann/arctic_shift/blob/master/README.md. We collected and pulled comments and posts from 12/22/22 to 9/18/26 for a total of 545k posts and 7.44M comments. These are saved to `data/r_chatGPT_posts_2022-12-22_2026-9-18.jsonl` (545k) and `data/r_chatGPT_comments_2022-12-22_2026-9-18.jsonl` (7.44M) respectively.

## Generate Candidate Corpus (Input to Embeddings)
Steps 1 and 2 both read the raw dumps and are independent of each other; steps 3-6 need both.

### 1. Metadata sidecars: `build_metadata.py`
Creates `data/meta_comments.parquet` (id, link_id (post id), parent_id (parent comment `t1_` or post `t3_`), author, created_utc, score, controversiality, permalink) and `data/meta_posts.parquet` (id, author, created_utc, score, num_comments, upvote_ratio, title, permalink, selftext_status (deleted / removed / empty / text), post_kind (text / image / video / link, from the post url), link_flair_text) from the original datasets mentioned above. Streams the 14 GB comments dump in ~10 s.

The text files from step 2 only carry what the embedding model needs; everything else is joined from these sidecars by `id`. The pipeline itself uses `link_id`, `parent_id`, `score`, `permalink` (step 6). The remaining columns (`controversiality`, `upvote_ratio`, `selftext_status`, `post_kind`, `link_flair_text`) are not used yet: they are covariates for SAE feature selection (e.g. features that separate controversial from uncontroversial comments, or screenshot threads from text threads). Note: the raw comments dump contains ~5k duplicate ids, and comment and post ids share one namespace, so joins are done per type after `drop_duplicates("id")`.

### 2. Filter dataset: `input_to_embeddings.py`
Filter the above dataset by removing the following from the dataset:
1. Empty text or literal '[deleted]'/'[removed]' bodies
2. Duplicate ids: keep first occurrence only
3. Duplicate exact text context: keep first occurrence only
4. Bot authors: `AutoModerator`
5. Token length above `max-tokens` (default: 2048)
6. Low-word-count: remove posts/comments with fewer than `min-words` (default: 10)
7. Low-engagement posts: remove posts with fewer than `min-post-comments` (default: 2)
8. Optionally remove: posts/comments flagged with `"_meta.removal_type` (enable with `--drop-removal-type`)

This results in *4,042,661* (from 7.44M) comments and *177,145* (from 545k) posts saved to `data/text_comments_for_embed_2022-12-22_2026-9-18.jsonl` and `data/text_posts_for_embed_2022-12-22_2026-9-18.jsonl` respectively (~251M + ~31M ~= 282M tokens).

### 3. Find candidate terms especially present in sycophantic discourse (by log-odds): `build_lexicon.py`
This grabs the terms that are especially present in the positive (`text_comments_sycophan_for_embed.jsonl`, filtered on "sycophan" keyword, n=18,762 comments after dropping AutoModerator rows, which this older file still contains), but not present in background (every 20th comment of `text_comments_for_embed_2022-12-22_2026-9-18.jsonl`, unfiltered, n=202,130 comments) datasets. This gives us the terms that are related to sycophancy as it gets the terms that often are in relation to sycophancy-related discourse.
1. Get unigram and bigram terms from each dataset - removing common `stop` words (eg: the, a, this). Count the number of documents each unigram/bigram appears in (document frequency, so one long comment cannot dominate).
2. Calculate the log_odds for the term counts for positive and background generated from 1, using the informative-Dirichlet-prior estimator of Monroe, Colaresi & Quinn (2008). A high z-score corresponds to terms that are especially present in the positive dataset compared to the background dataset. Delta vs z: *delta* is the raw log-odds difference (the effect size), which is largest and least reliable for rare terms - a word seen 3 times in the positive set and never in the background has an enormous delta. *z* divides delta by its standard error (roughly `sqrt(1/count_pos + 1/count_bg)`), so a term needs both a large difference *and* enough evidence to rank highly. We rank by z.
3. Save the top (default: 300) terms sorted by z-score descending to `data/candidates/logodds_terms.csv`, with an `in_lexicon` column marking terms the current lexicon already matches. The top of the list mixes behaviour descriptors (sycophantic, flattery, glazing, agreeable, push back, validation, praise, psychosis, delusions) with topic words that co-occur with the discourse but do not describe it (personality, model, behavior, tone, custom instructions, rollback); only the former are candidates for the lexicon.

### 4. Curate the terms to filter/select candidate posts/comments: `lexicon.py`
The lexicon is a hand-curated list of regex fragments in three tiers: tier1 (names the behavior: eg: sycophan, glaze, yes man), tier2 (phrases that describe the behavior: "tells you what you want to hear", "agreeable"), tier3 (adjacent discourse: "psychosis", "delusion", "emotionally dependent"). The tier is stored as a label on each hit, not used as a filter, so the SAE can be trained with or without tier 3.

The list was built in two versions, and each term records which one it entered in (`*_V1` / `*_V2` lists in `lexicon.py`), so the two corpora can be compared from one embedding run:
- **v1**: the hand-written list (jargon plus phrases from the earlier manual reading). 30,249 hits.
- **v2**: v1 plus terms selected from the step-3 ranking. A term was added if (a) it describes the behaviour rather than the topic, (b) a random sample of its *new* hits (docs not already caught by v1; 6-7 read per term) was mostly about the chatbot's agreeableness, and (c) broad verbs were constrained to the chatbot-to-user sense (e.g. `validates (me|you|everything...)`, `compliments me`, not bare `validat*`). Rejected: "echo chamber" (~4k hits, mostly about Reddit/politics), bare "praise" (~2/6 on-topic), bare "validat*" (16k hits, mostly other senses), and phrases like "great question" that are sincere as often as mocking. 61,509 hits.

`docs/lexicon.md` lists every term with its hit count, the spot-check results, and the rejected terms.

### 5. Filter corpus by terms from step 4: `filter_lexicon.py`
Takes in `data/text_comments_for_embed_2022-12-22_2026-9-18.jsonl` and `data/text_posts_for_embed_2022-12-22_2026-9-18.jsonl` and counts the number of "hits" (each time the "text" in the input file matches the regex from `lexicon.py`). Saves the text that matched the lexicon to `data/candidates/lexicon_hits_for_embed.jsonl` (id, text, is_post, token_length) and metadata `data/candidates/lexicon_hits.parquet` (id, is_post, token_length, tier (lowest tier matched), terms (all terms matched by text), in_v1 (whether the v1 lexicon alone would have matched)).
Results in *61,509* comments and posts that match the lexicon (57,402 comments + 4,107 posts): 23,907 tier 1, 19,826 tier 2 only, 17,776 tier 3 only; 30,249 of them match v1. Takes ~11 min (the regex is run on the raw line first so only candidate lines are JSON-parsed).

### 6. Assemble the candidate corpus: lexicon hits (step 5) + direct replies to those hits: `build_candidates.py`
From the corpus from step 5, we additionally add comments that are a direct reply to those comments/posts (direct reply if its parent_id points to the lexicon-hit comment or lexicon-hit post), found through `meta_comments.parquet`. 
We cap the max replies per thread (`--cap`, default: 200, highest score kept) so that a few megathreads do not dominate the corpus; lexicon hits themselves are never capped. Replies are labelled `route = 'reply'` (lexicon hits: `route = 'lexicon'`) and carry their parent's `tier` and `in_v1`, so the reply population can be included or excluded later.

Accounting for the current run: 89,624 direct replies found -> 7,152 dropped by the cap (28 threads; the largest was a "what animal does ChatGPT think you are" thread with 1,430 replies whose post text matched "flattering") -> 24,750 had no row in the step-2 text files (deleted, under 10 words, or bot) -> **57,618 replies kept**. Total corpus: **119,127 docs** (61,509 lexicon hits + 57,618 replies), 14.8M tokens, 20,342 threads.

Saves the following output (in `data/candidates/`):
1. candidates.parquet: id, is_post, route, tier, terms, in_v1, parent_tier, parent_in_v1, link_id, parent_id, created_utc, score
2. candidates_for_embed.jsonl: id, text, is_post, token_length
3. capped_threads.csv: threads over cap (link_id, permalink, n_replies, n_kept, n_dropped)
4. capped_dropped.parquet: every reply dropped by the cap (id, link_id, parent_id, score)

Because every row keeps its provenance, one embedding run of `candidates_for_embed.jsonl` supports four training corpora, selected by id from `candidates.parquet`:

| corpus | selection | docs |
|---|---|---|
| v1 | `route == 'lexicon' & in_v1` | 30,249 |
| v2 | `route == 'lexicon'` | 61,509 |
| v1 + replies | `(route == 'lexicon' & in_v1) \| (route == 'reply' & parent_in_v1)` | 58,559 |
| v2 + replies | all rows | 119,127 |

(add `& tier < 3` to drop the adjacent tier; the cap was applied once over all replies, so the v1 + replies subset is slightly stricter than a standalone v1 run would be.)

## Embedding Generation: `src/embedding_generation/embed_documents.py`
We use `nvidia/llama-embed-nemotron-8b` (official bf16 weights via sentence-transformers, no instruction prefix for documents, L2-normalised) to embed the documents generated from the above step. This step should use Google Colab/GPU (A100: ~30 min for the 119k docs). Ensure that your output is saved to drive so that you don't lose the file when Runtime is disconnected: the script writes one `.npy` shard per 2,000 docs next to the output path and skips finished shards on re-run, so a disconnect only costs the current shard.

In Colab, after uploading `candidates_for_embed.jsonl` and `embed_documents.py` to Drive:
1. `!pip install -q -U sentence-transformers`
2. `from huggingface_hub import login; login()` - the model is gated; accept its licence on Hugging Face first
3. `from google.colab import drive; drive.mount('/content/drive')`
4. `!python /content/drive/MyDrive/<folder>/embed_documents.py -i /content/drive/MyDrive/<folder>/candidates_for_embed.jsonl -o /content/drive/MyDrive/<folder>/embeddings_candidates_nemotron.npz`

Optional determinism check before the full run: run with `-n 200 -s 100` twice into two output paths; the two `.npz` files should be bit-identical.

Output: `embeddings_candidates_nemotron.npz` with `ids` (119,127, same order as the input JSONL) and `embeddings` (119,127 x 4096, float16). Copied to `data/embeddings/` for training. Only `ids` and `embeddings` are stored; everything else is joined from `candidates.parquet`.

## SAE Training and Interpretation: `src/scripts/`
- `train_sae.py -e <npz> -n <M> -k <K> -c <checkpoint_dir> [-p <matryoshka prefix lengths>]` trains a top-K SAE (HypotheSAEs implementation in `third_party/sae.py`) on an 80/20 split of the embeddings and reports the top features by mean activation.
- `interpret_sae.py -e <npz> -i <text jsonl> -n <M> -k <K> -c <checkpoint_dir>` labels the top features with an LLM (`--interpreter-model`, `--annotator-model`; needs `OPENAI_API_KEY`).
- Selecting one of the four corpora above from the single `.npz` is not implemented yet: `train_sae.py` currently trains on every row of the file. The first planned run is v2 (`route == 'lexicon'`).