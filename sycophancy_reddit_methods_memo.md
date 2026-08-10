# Methods Memo: Public View of Sycophancy in LLMs — Reddit/SAE Pipeline

Prepared as a skeptical review of the current data + proposed embedding/SAE approach. Data checked directly (see "Data audit" below); three source papers read in full.

---

## 1. Data audit — two problems to fix before anything else

**`chat_gpt_comments_sycophan.jsonl` is broken JSONL.** It was written pretty-printed (multi-line, indented) instead of one-JSON-object-per-line. `wc -l` reports 69,697 — that's *lines of text*, not records. Parsed correctly (concatenated-object parse), there are **905 actual comment objects**, all of which do contain "sycophan" in the body — so the filter logic was correct, only the file writer was wrong. Any script doing `for line in f: json.loads(line)` or `pd.read_json(path, lines=True)` on this file will either throw or silently return 0 rows. Reformat to true JSONL before doing anything else — one-line fix (`json.dumps(obj) + '\n'` per record).

**True corpus size is ~1,000 texts, not tens of thousands.** 98 posts + 905 comments. Base rates against the unfiltered corpus: 98/32,757 posts (0.30%) and 905/645,101 comments (0.14%) mention "sycophan." That's a small, keyword-selected sample — treat every downstream claim as exploratory/descriptive, not representative, and say so explicitly in the eventual writeup (the r/ChatGPT paper below makes the same caveat about Reddit skewing young/male/white/educated — worth citing).

## 2. The bigger methodological problem: filtering *before* concept discovery defeats the purpose

Running embeddings/SAE on only the "sycophan"-filtered subset will surface sub-topics *within* text that already uses the word. It cannot, by construction, find people talking about the same behaviors without naming them — which is literally the open question already in the audit doc ("what % of the public understands sycophancy as a term vs. just notices the behavior"). Filter-then-cluster answers "how do people who say 'sycophancy' talk about it." It doesn't answer "who's describing sycophantic behavior without the word."

Fix: run concept discovery on the **full, unfiltered** r/ChatGPT corpus (32,757 posts, optionally + the 645K comments), and use "contains sycophan-root" as a *target label* to find which discovered concepts predict that label — rather than using the keyword as a corpus filter. This is exactly the design of both papers below (HypotheSAEs' target-variable framing; the r/ChatGPT paper's SAE-over-everything approach). It also directly serves the "newfound examples not captured by the expert taxonomy" goal, since concepts that predict the label without containing the word are the interesting ones.

Two-track plan: today, do the fast version on the small filtered set to get a working pipeline and a first pass at concepts. Treat the full-corpus run as the next iteration, not today's deliverable.

## 3. Source summaries

### Ye et al., "What Counts as AI Sycophancy?" (2605.21778)
70-paper literature review + 106-expert survey. Produces a 2×2 taxonomy: **Referent** (does the behavior respond to the user's *Position* — claims/opinions — or to the user *as a Person* — traits/emotions?) × **Explicitness** (explicit vs. implicit — framing, omission, tone). Sub-referents: Verifiable/Subjective under Position, Traits/Emotions under Person. Key empirical finding: experts agree Position-sycophancy is sycophantic regardless of explicitness, but for Person-sycophancy, only *explicit* forms (flattery, unwarranted praise) are reliably recognized — implicit deference/tone-softening is not. 94.3% of experts agree sycophancy is a real problem; single-rater reliability on *which behaviors count* is low (ICC2 = .184) — i.e., even experts don't agree on the construct.

**Applicable:** use this taxonomy as the target ontology. After you have discovered SAE concepts, hand-code (or LLM-code) each one into a taxonomy cell. This directly operationalizes goal 4.4 in the audit doc ("newfound examples/groupings vs. expert view") — anything that doesn't map cleanly onto a cell, or that shows laypeople conflating cells experts treat as separate (e.g., not distinguishing implicit Person-sycophancy from explicit), is a genuine finding. Also gives you a testable hypothesis: do Reddit users, like experts, fail to flag implicit Person-sycophancy (warmth, avoidance of critique) as "sycophancy" even when they describe it?

### Dai et al., "Three Years of r/ChatGPT" (2606.05750)
This is the closest thing to a working blueprint for your project. Pipeline: concatenate post title+body → embed with `text-embedding-3` → train a top-K SAE (K=4, M=128, later M=64 for a real-time variant) on the embeddings, sample-weighted by engagement (log of upvotes+comments) → interpret each feature with `gpt-4.1-mini` using the interpretation prompts from Movva et al. (their appendix G) → keep the best of 3 candidate interpretations by F1 → LLM-label every post (majority vote of 3 calls) for each surviving feature → group features into "families" by co-occurrence and by temporal-trajectory similarity → fit piecewise-linear trend models against a timeline of product releases to find features that are *reactive* to specific events vs. drifting over the whole period.

Substantive finding structure worth imitating directly: they found two feature families — "domestication" (routine use) and "emotional engagement" (therapy, companionship, naming the AI, romantic framing) — and showed *therapy* and *companion* are correlated but semantically and behaviorally distinct (different co-occurring features, different vocabulary via log-odds word analysis, different sensitivity to model updates). Your audit doc's "downstream/interested topics" goal (self-therapy etc.) and "evolvement over time" goal map directly onto this method — you'd be doing the same feature-family + event-changepoint analysis but centered on sycophancy-related features instead of emotional-engagement ones.

Also directly useful: their real-time monitoring add-on (PuLSE) isn't necessary for you today, but the retrospective method (Section 3) is fully sufficient and is essentially "HypotheSAEs, unsupervised, plus a changepoint test against a release-date timeline."

**Caveat this paper makes about itself, worth repeating in yours:** Reddit users are ~1.9% of all ChatGPT traffic even for emotionally-loaded topics per OpenAI's own usage report (Chatterji et al. 2025) — high visibility on Reddit doesn't mean high prevalence among all users. Frame your findings as "what's salient enough to post about," not "what fraction of users feel this way."

### Peng, Movva et al., "Position: Use SAEs to Discover Unknowns" (2506.23845)
The conceptual argument you need before writing any code: SAEs underperform simple baselines (logistic regression, prompting) when used to **detect a known concept** in text or **steer** a model toward one — this is the "SAEs don't work" narrative your friend may have heard. But SAEs *do* outperform baselines at **discovering unknown concepts** — i.e., enumerating a set of candidate concepts from data with no target pre-specified, or ones best correlated with a target variable (their own hypothesis-generation work, Movva et al. 2025, is the strongest positive result cited). Practical implication for you: don't use an SAE as your "is this text about sycophancy" classifier (use keyword match or a prompted LLM for that — a known-concept task, where simple baselines win). Do use an SAE to find *what the sycophancy-adjacent concepts actually are*, which is an unknown-concept task.

Section 2 of this paper is a self-contained, non-technical SAE primer (autoencoder → sparsity constraint → interpretable neurons → auto-interpretation via LLM) — this is the fastest on-ramp for someone who's worked with embeddings but never with SAEs. Read this before touching code.

## 4. The tool: don't write an SAE from scratch

`pip install hypothesaes` — built by the same authors as the position paper and the hypothesis-generation paper it cites (Movva, Peng, Garg, Kleinberg, Pierson). This is not "one option among several," it's the reference implementation the r/ChatGPT paper itself is built on top of. https://github.com/rmovva/HypotheSAEs

Five-step pipeline, all in the package: (1) embed texts (OpenAI or local sentence-transformers), (2) train top-K SAE on embeddings, (3) select the SAE neurons most predictive of a target variable (correlation / LASSO / separation score), (4) LLM-interpret each selected neuron into a natural-language concept, (5) LLM-annotate a heldout set to validate that the concept actually predicts the target.

Concrete numbers relevant to your data size, from their own hyperparameter guidance:
- ~1,000 examples (your filtered set) → `M=64, K=4`
- ~10,000 examples → `M=256, K=8`
- ~100,000 examples (full posts corpus + a comment sample) → `M=1024, K=8`

Cost/speed reference they give: 20K texts, full pipeline, ~2 minutes and ~$0.40 in API calls. Your filtered set (1,003 texts) will be faster and cheaper than that. No GPU needed if you use OpenAI for embeddings/interpretation — CPU is fine for SAE training at this scale.

One thing to decide before running it: **what's your target variable?** If you run the small filtered set, you don't have an obvious one yet (everything already contains "sycophan") — so run it unsupervised (interpret the top-activating neurons directly, no target) the way the r/ChatGPT paper does for its retrospective analysis. If you run the full corpus, the target is "mentions sycophan-root" (binary) — this is the run that actually produces the "concepts predictive of sycophancy discourse" result you want.

## 5. Concrete outline for today

1. **Fix the JSONL bug** and rebuild a clean combined file: `{id, type: post|comment, text, created_utc, score, num_comments}`. Dedupe on id. (~15 min)
2. **Sanity EDA before spending API budget**: volume of posts/comments over time, score distribution, text length distribution. Confirms nothing else is broken. (~15–20 min)
3. **Embed the filtered set** (1,003 texts) with `text-embedding-3-small`. Seconds to a couple minutes, trivial cost. (~15 min)
4. **Checkpoint before SAE**: UMAP + HDBSCAN (or even just k-means) on the raw embeddings. This alone is a legitimate "learned something today" result and a sanity check that the embedding space has structure before you add SAE complexity on top. (~30 min)
5. **Run HypotheSAEs unsupervised** on the same 1,003 embeddings (`M=64, K=4`), interpret the top ~20 neurons with the interpreter LLM. Output: first list of candidate concepts from the sycophan-labeled subset. (~45–60 min including reading the interpretations)
6. **Map each discovered concept onto the taxonomy paper's grid** (Referent × Explicitness). Note anything that doesn't fit a cell. (~20 min)

That's a realistic full pipeline, small-scale, done today. Full-corpus run (unfiltered posts + target-variable HypotheSAEs, the version that actually answers the "implicit sycophancy without the word" question) is next session's work — don't try to compress both into one day.

## 6. Other things to flag now, before they become unfounded claims later

- **No demographic metadata exists in Reddit data.** Age, occupation, etc. are not fields — any claim about "who" holds which view requires inferring from self-disclosure text ("as a therapist I...") via LLM extraction, which is noisy and, since it's inferring attributes of anonymous individuals, worth thinking about from a research-ethics angle before it goes in the paper (this data is public but individual users didn't consent to demographic profiling).
- **Reddit-vs-population gap**: cite the r/ChatGPT paper's own numbers (Reddit skews young/male/white/educated; ~1.9% of *all* ChatGPT traffic is emotionally-toned per OpenAI's usage report) to bound claims about "the public."
- **SAE feature interpretation isn't free of noise.** Use the `n_candidate_interpretations=3` + F1-scoring option in HypotheSAEs rather than trusting a single LLM-generated label per neuron — the position paper and the r/ChatGPT paper both flag this as necessary, not optional.

## Reading list, ranked by what to actually do with limited time today

1. Position paper §2 (SAE primer) — 15 min, non-technical, read first.
2. HypotheSAEs README + `quickstart.ipynb` — hands-on, same task shape as yours (predict a target from text via SAE concepts), ~30–60 min including running it once on their toy Yelp example before touching your own data.
3. Taxonomy paper (2605.21778) — read Table 1 (the taxonomy grid) and the expert-survey results section; skip the corporate/legislative governance discussion for now, it's not needed to get started.
4. r/ChatGPT paper (2606.05750) — read Section 2 (method) and Section 3 (findings) closely; this is your analysis template. Section 4 (PuLSE real-time monitoring) is not needed today.
