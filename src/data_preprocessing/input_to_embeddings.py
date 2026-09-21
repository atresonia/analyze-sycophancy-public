"""
Converts Reddit comment/post input data (JSONL file)
to a JSONL file with text, is_post, and token_length
for compatibility with embedding models.

We will use the nemotron embedding model to encode the texts into a list of embeddings.

Filtering applied:
- Empty text, or literal '[deleted]'/'[removed]' bodies.
- Duplicate ids (keeps the first occurrence; deduped across all input files when merging).
- Duplicate exact text content, after normalization (keeps the first occurrence).
- Bot authors (e.g. AutoModerator), case-insensitive.
- Token length above --max-tokens (embedding-model context compatibility).
- Low-word-count spam: any row (post or comment) with fewer than --min-words words in its text
  is dropped, per the filter in https://arxiv.org/pdf/2606.05750.
- Low-engagement posts: posts with fewer than --min-post-comments comments are dropped
  (comments are unaffected by this one, since they're included as their own rows regardless
  of their parent post's engagement).
- Optionally, posts/comments flagged with a moderation "_meta.removal_type" (off by default,
  since these can still carry valid visible text worth keeping; enable with --drop-removal-type).

Usage:
- convert posts file to embedding-friendly (default): python input_to_embeddings.py
- convert comments file to embedding-friendly: python input_to_embeddings.py -i data/r_chatGPT_comments.jsonl -o data/text_comments_for_embed.jsonl
- convert combined file to embedding-friendly: python input_to_embeddings.py -i data/r_chatGPT_comments.jsonl data/r_chatGPT_posts.jsonl -o data/text_combined_for_embed.jsonl
"""

import argparse
import hashlib
import json
from transformers import AutoTokenizer
import os


MODEL_NAME = "mlx-community/Qwen3-Embedding-8B-4bit-DWQ"
DEFAULT_MAX_TOKEN_LENGTH = 2048  # matches MAX_LENGTH in embed_documents_memory_constrained.py
DEFAULT_MIN_WORDS = 10
DEFAULT_MIN_POST_COMMENTS = 2
DEFAULT_EXCLUDED_AUTHORS = ["AutoModerator"]


def calculate_token_length(tokenizer: AutoTokenizer, text: str) -> int:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def normalize_text_for_hash(text: str) -> str:
    return " ".join(text.strip().lower().split())


def get_text_from_object(obj: dict, drop_removal_type: bool = False) -> tuple[str | None, bool]:
    is_post = "title" in obj or "selftext" in obj
    if drop_removal_type and "_meta" in obj and "removal_type" in obj["_meta"]:
        return None, is_post
    if is_post:
        title = obj.get('title', '').strip()
        selftext = obj.get('selftext', '').strip()
        if selftext in ('[deleted], [removed]'):
            return None, is_post
        text = f"{title}\n\n{selftext}" if selftext else title
    else:
        text = obj.get('body', '').strip()
    if not text or text in ('[deleted]', '[removed]'):
        return None, is_post
    return text, is_post


def convert_input_files_to_json_object(
    input_files: list[str],
    tokenizer: AutoTokenizer,
    output_file: str,
    max_tokens: int = DEFAULT_MAX_TOKEN_LENGTH,
    min_words: int = DEFAULT_MIN_WORDS,
    min_post_comments: int = DEFAULT_MIN_POST_COMMENTS,
    excluded_authors: list[str] | None = None,
    drop_removal_type: bool = False,
) -> None:
    """Takes in one or more input files, merges them (if multiple), and converts to an output file
    with keys 'id', 'text', 'is_post', and 'token_length'. Duplicate ids and duplicate exact text
    (including duplicates across input files) are skipped, keeping only the first occurrence."""
    excluded_authors_lower = {a.lower() for a in (excluded_authors or [])}
    counts = {
        "input_lines": 0,
        "dropped_empty_or_removal_type": 0,
        "dropped_duplicate_id": 0,
        "dropped_duplicate_text": 0,
        "dropped_bot_author": 0,
        "dropped_token_length": 0,
        "dropped_low_word_count": 0,
        "dropped_low_engagement_post": 0,
        "missing_id": 0,
        "posts": 0,
        "comments": 0,
        "output_rows": 0,
    }
    seen_ids = set()
    seen_text_hashes = set()
    with open(output_file, 'w', encoding='utf-8') as target:
        for file_path in input_files:
            convert_input_file_to_list(
                file_path, tokenizer, target, seen_ids, seen_text_hashes,
                excluded_authors_lower, drop_removal_type, max_tokens,
                min_words, min_post_comments, counts,
            )
    print(f"Processed {counts['input_lines']} lines")
    print(f"Dropped {counts['dropped_empty_or_removal_type']} empty lines or removed/deleted posts/comments")
    print(f"Dropped {counts['dropped_bot_author']} rows from excluded (bot) authors")
    print(f"Dropped {counts['dropped_duplicate_id']} duplicate ids")
    print(f"Dropped {counts['dropped_duplicate_text']} duplicate texts")
    print(f"Dropped {counts['dropped_low_word_count']} rows with fewer than {min_words} words")
    print(f"Dropped {counts['dropped_low_engagement_post']} posts with fewer than {min_post_comments} comments")
    print(f"Dropped {counts['dropped_token_length']} rows over {max_tokens} tokens")
    if counts["missing_id"]:
        print(f"Warning: {counts['missing_id']} rows had no 'id' field and could not be deduped by id")
    print(f"Found {counts['posts']} posts and {counts['comments']} comments")
    print(f"Wrote {counts['output_rows']} rows to {output_file}")


def convert_input_file_to_list(
    input_file: str,
    tokenizer: AutoTokenizer,
    target,
    seen_ids: set,
    seen_text_hashes: set,
    excluded_authors_lower: set,
    drop_removal_type: bool,
    max_tokens: int,
    min_words: int,
    min_post_comments: int,
    counts: dict,
) -> None:
    """
    Takes in a JSONL file and extracts the text from each object.
    Writes to the already-open target file with keys 'id', 'text', 'is_post', and 'token_length'.
    If the object has a 'body' field (is a comment), the text is written to the file with is_post set to False.
    If the object has a 'selftext' or 'title' field (is a post), the text is combined and written to the file with is_post set to True.
    Skips objects whose id or exact (normalized) text has already been seen, whose author is excluded,
    whose token length exceeds max_tokens, whose text has too few words, or (for posts only) whose
    parent post has too few comments.
    """
    with open(input_file, 'r', encoding='utf-8') as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            counts["input_lines"] += 1
            obj = json.loads(line)
            obj_id = obj.get('id')
            if obj_id is None:
                counts["missing_id"] += 1
            if obj_id is not None and obj_id in seen_ids:
                counts["dropped_duplicate_id"] += 1
                continue

            author = obj.get('author')
            if author is not None and author.lower() in excluded_authors_lower:
                counts["dropped_bot_author"] += 1
                continue

            text, is_post = get_text_from_object(obj, drop_removal_type=drop_removal_type)
            if not text:
                counts["dropped_empty_or_removal_type"] += 1
                continue

            if len(text.split()) < min_words:
                counts["dropped_low_word_count"] += 1
                continue

            if is_post and obj.get('num_comments', 0) < min_post_comments:
                counts["dropped_low_engagement_post"] += 1
                continue

            text_hash = hashlib.md5(normalize_text_for_hash(text).encode('utf-8')).hexdigest()
            if text_hash in seen_text_hashes:
                counts["dropped_duplicate_text"] += 1
                continue

            token_length = calculate_token_length(tokenizer, text)
            if token_length > max_tokens:
                counts["dropped_token_length"] += 1
                continue

            if obj_id is not None:
                seen_ids.add(obj_id)
            seen_text_hashes.add(text_hash)

            if is_post:
                counts["posts"] += 1
            else:
                counts["comments"] += 1
            counts["output_rows"] += 1
            target.write(json.dumps({
                'id': obj_id,
                'text': text,
                'is_post': is_post,
                'token_length': token_length,
            }, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Convert Reddit subthread data to suitable format for embeddings generation")
    parser.add_argument("-i", "--input_files", nargs='+', default=["data/r_chatGPT_posts.jsonl"],
    help="Either provide posts, comments, or both file JSONL names. Takes at most two values.")
    parser.add_argument("-o", "--output_file", type=str, default="data/text_posts_for_embed.jsonl")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKEN_LENGTH,
        help="Drop rows with more than this many tokens.")
    parser.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS,
        help="Drop posts or comments with fewer than this many words in their text (spam filter).")
    parser.add_argument("--min-post-comments", type=int, default=DEFAULT_MIN_POST_COMMENTS,
        help="Drop posts with fewer than this many comments (spam filter; comments unaffected).")
    parser.add_argument("--exclude-authors", nargs='*', default=DEFAULT_EXCLUDED_AUTHORS,
        help="Author names to exclude (case-insensitive), e.g. bot accounts.")
    parser.add_argument("--drop-removal-type", action="store_true",
        help="Also drop posts/comments flagged with '_meta.removal_type' (kept by default, "
             "since they can still carry valid visible text).")

    args = parser.parse_args()
    if len(args.input_files) > 2:
        parser.error('--input_files accepts between 1 and 2 arguments only.')

    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    convert_input_files_to_json_object(
        args.input_files, tokenizer, args.output_file,
        max_tokens=args.max_tokens,
        min_words=args.min_words,
        min_post_comments=args.min_post_comments,
        excluded_authors=args.exclude_authors,
        drop_removal_type=args.drop_removal_type,
    )
    print(f"Wrote texts to {args.output_file}")


if __name__ == '__main__':
    main()
