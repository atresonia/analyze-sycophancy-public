# Sycophancy lexicon

Record of the lexicon in `src/data_preprocessing/lexicon.py`: every term, the version it entered in, how many documents it hits in the full r/ChatGPT text files (4,042,661 comments + 177,145 posts, 2022-12-22 to 2026-09-18), how the v2 additions were checked, and what was rejected. Hit counts are documents matching the term; a document can match several terms.

Matching is case-insensitive regex over the post/comment text. A document's `tier` is the lowest tier among its matched terms; `in_v1` is true if any v1 term matched.

## Tier 1 - names the behaviour

| term (regex) | version | hits | note |
|---|---|---|---|
| `sycophan` | v1 | 7,508 | |
| `glaz(e\|ing\|ed\|er)s?\b(?! over)` | v1 | 5,697 | "glazes over the details" excluded |
| `flatter` | v1 | 5,444 | also "flattering", "flattery" |
| `yes.?m[ae]n` | v1 | 3,119 | |
| `boot.?lick` | v1 | 756 | |
| `ass.?kiss` | v1 | 577 | |
| `suck.?up` | v1 | 393 | |
| `obsequious` | v1 | 245 | |
| `kiss.?ass` | v1 | 228 | |
| `brown.?nos` | v1 | 171 | |
| `bajula` | v1 | 10 | Portuguese (bajulação) |
| `s[ií].?buey` | v1 | 1 | Spanish |
| `\bpander` | v2 | 982 | |
| `\bfawn` | v2 | 306 | |

## Tier 2 - describes the behaviour without naming it

| term (regex) | version | hits | note |
|---|---|---|---|
| `sugar.?coat` | v1 | 2,448 | almost all "no sugarcoating" custom instructions (anti-sycophancy prompting) |
| `tells? (me\|you\|people\|users?) what (i\|you\|they) want to hear` | v1 | 1,218 | |
| `agrees? with everything` | v1 | 868 | |
| `people.?pleas` | v1 | 818 | |
| `cheerleader` | v1 | 668 | ~half are "AI as my cheerleader" (wants-validation framing) |
| `(too\|overly\|so) agreeable` | v1 | 594 | |
| `blow(ing\|s)? smoke` | v1 | 517 | |
| `hype ?man` | v1 | 419 | |
| `strok(e\|es\|ing) (my\|your\|the user'?s?) ego` | v1 | 244 | |
| `just agrees` | v1 | 226 | |
| `dick.?rid` | v1 | 207 | |
| `pat on the (back\|head)` | v1 | 192 | |
| `validat(es\|ion) (everything\|machine)` | v1 | 144 | |
| `feed(ing\|s)? (my\|your) ego` | v1 | 60 | |
| `push.?back` | v2 | 5,084 | "it never pushes back" |
| `agreeable` | v2 | 3,028 | supersedes the v1 `(too\|overly\|so) agreeable`; also matches agreeableness, disagreeable |
| `brutal(ly)? honest` | v2 | 2,144 | prompting theme, like sugarcoat |
| `(seek\|...\|cheap) validation` | v2 | 1,492 | full adjective list in `lexicon.py` |
| `validat(es\|ing\|ed) <object>` | v2 | 1,457 | `<object>` = me / you / them / everything / my / your / the user's ... |
| `agree(s\|ing)? with (everything\|whatever\|anything)` | v2 | 1,444 | supersedes the v1 `agrees? with everything` |
| `compliment(s\|ing\|ed)? <object>` | v2 | 879 | |
| `(?<!gender[- ])affirm(s\|ing\|ed)? <object>` | v2 | 776 | "gender-affirming" excluded |
| `validation (machine\|seeking\|bot\|loop\|engine)` | v2 | 190 | |
| `(constant\|endless\|empty\|unconditional\|blind)(ly)? affirm` | v2 | 108 | |
| `engage warmly\|warmly yet\|grounded honesty` | v2 | 53 | OpenAI's April-2025 system-prompt wording, quoted in rollback threads |
| `(constant\|...\|random) compliments` | v2 | 35 | |

## Tier 3 - adjacent discourse (harm rather than behaviour)

| term (regex) | version | hits | note |
|---|---|---|---|
| `delusion` | v2 | 12,521 | delusion / delusions / delusional |
| `psychosis\|psychotic` | v2 | 7,631 | "AI psychosis" threads |
| `emotional(ly)? dependen` | v2 | 788 | |

Kept as a separate tier because it is a different corpus boundary: these documents are about the consequences of validation-seeking use, not about the model's agreeableness. Spot check of new-only hits: 5 of 7 in the AI-psychosis sense, 2 of 7 generic "you're delusional" insults.

## How the v2 terms were chosen

1. `build_lexicon.py` ranked unigrams and bigrams by log-odds z-score (Monroe, Colaresi & Quinn 2008, informative Dirichlet prior) between the 17k "sycophan" comment set (18,762 docs after removing AutoModerator rows) and a 1-in-20 sample of all comments (202,130 docs). Output: `data/candidates/logodds_terms.csv`. The v1 terms all ranked highly with enrichment ratios of 9-65x (sycophantic 66 z, flattery 21x, glazing 16x, yes man 17x, cheerleader 16x, people pleasing 22x), which is the check that the hand list was not idiosyncratic.
2. From the top ~400 terms, behaviour descriptors were separated from topic words (personality, model, tone, custom instructions, rollback, therapist). Topic words are left for the SAE to find.
3. For each candidate the number of *new* documents it would add (not already matched by v1) was counted, and 6-7 of those new documents were read at random. Terms whose new hits were mostly about the chatbot's agreeableness were added; broad verbs were constrained to a chatbot-to-user object and re-checked.

Spot-check results (n = 6-7 new-only documents per term; these are ranking evidence, not precision measurements - the 95% interval on each is roughly +/- 35 points):

| candidate | new docs | on-topic in sample | decision |
|---|---|---|---|
| agreeable | 2,025 | 5/7 | added |
| push back | 4,460 | 5/7 | added |
| validat* (bare) | 16,186 | 3/7 | rejected - "input validation", "invalidate", social-media validation |
| validation (constrained) | 2,595 | 5/6 | added |
| praise (bare) | 5,559 | 2/7 | rejected - "praised her beauty", "praise be" |
| praise (constrained) | 718 | 2/6 | rejected - still mostly human praise |
| affirm* (bare) | 4,309 | 3/7 | rejected - "affirm my suspicions", "neurodiversity-affirming" |
| affirm (constrained) | 738 | 5/6 | added |
| compliment (bare) | 4,351 | 4/7 | rejected - "complement my career", "take it as a compliment" |
| compliment (constrained) | 783 | 4/6 | added |
| psychosis / delusion / emotional dependency | 16,926 | 5/7 | added as tier 3 |
| fawn, pander, brutally honest, agree with whatever/anything, OpenAI system-prompt quote | 239 / 860 / 1,673 / 463 / 13 | specific, not sampled | added |

## Rejected terms

| term | reason |
|---|---|
| echo chamber | ~4,000 hits, 1-2 of 6 sampled were about the chatbot; the rest Reddit/politics/social media |
| praise (any form) | see above |
| validat* / affirm* / compliment (bare forms) | see above; constrained forms kept |
| great question, you're absolutely right, nail on the head | sincere as often as mocking; would need thread context to disambiguate |
| hedging, condescending, patronizing, gaslight | different behaviour (over-correction / lecturing), 4-13x enriched but not sycophancy |
| mirror, empathy, honest, positive, warm, nice | 2-4x enriched; discourse themes rather than descriptors, too broad |
| therapist, therapy, mental health, friend, lonely, companion | emotional-support population; too broad for a lexicon, left for the SAE |
| personality, tone, custom instructions, system prompt, rollback, rlhf, guardrails | topic/context words, not behaviour |

## Result

| | comments | posts | total |
|---|---|---|---|
| tier 1 (jargon) | 22,128 | 1,779 | 23,907 |
| tier 2 only (phrase) | 18,364 | 1,462 | 19,826 |
| tier 3 only (adjacent) | 16,910 | 866 | 17,776 |
| **all (v2)** | **57,402** | **4,107** | **61,509** |
| of which v1 | 27,983 | 2,266 | 30,249 |

Known limits: precision was spot-checked, not measured - a labelled sample of ~50 documents per term is the next step if per-term precision needs to be reported. Recall is bounded by the vocabulary: descriptions of the behaviour that use none of these terms are only reached through direct replies (`build_candidates.py`), not through the lexicon.
