"""
Converts Reddit comment/post input data (JSONL file) 
to a JSONL file with text, is_post, and token_length 
for compatibility with embedding models.

We will use the nemotron embedding model to encode the texts into a list of embeddings.

Usage:
- convert posts file to embedding-friendly (default): python input_to_embeddings.py
- convert comments file to embedding-friendly: python input_to_embeddings.py -i data/r_chatGPT_comments.jsonl -o data/text_comments_for_embed.jsonl
- convert combined file to embedding-friendly: python input_to_embeddings.py -i data/r_chatGPT_comments.jsonl data/r_chatGPT_posts.jsonl -o data/text_combined_for_embed.jsonl
"""

import argparse
import json
from transformers import AutoTokenizer
import os


CHUNK_THRESHOLD_WORDS = 500
MODEL_NAME = "mlx-community/Qwen3-Embedding-8B-4bit-DWQ"

def calculate_token_length(tokenizer: AutoTokenizer, text: str) -> int:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def get_text_from_object(obj: dict) -> tuple[str, bool]:
    # if "_meta"."removal_type" exists, we do not want to use it as part of our data
    is_post = "title" in obj or "selftext" in obj
    if "_meta" in obj and "removal_type" in obj["_meta"]:
        return None, is_post
    if is_post:
        title = obj.get('title', '').strip()
        selftext = obj.get('selftext', '').strip()
        text = f"{title}\n\n{selftext}" if selftext else title
    else:
        text = obj.get('body', '').strip()
    return text, is_post


def convert_input_files_to_json_object(input_files: list[str], tokenizer: AutoTokenizer, output_file: str) -> None:
    """Takes in one or more input files, merges them (if multiple), and converts to an output file 
    with keys 'text', 'is_post', and 'token_length."""
    n_input_lines = n_dropped_empty_or_removal_type = n_posts = n_comments = n_output_rows = 0
    for file_path in input_files:
        loop_input, loop_dropped, loop_posts, loop_comments, loop_out = convert_input_file_to_list(
            file_path, tokenizer, output_file)
        n_input_lines += loop_input
        n_dropped_empty_or_removal_type += loop_dropped
        n_posts += loop_posts
        n_comments += loop_comments
        n_output_rows += loop_out
    # write summary to bottom of file as a {} object
    # summary = {
    #     "n_input_lines": n_input_lines,
    #     "n_dropped_empty_or_removal_type": n_dropped_empty_or_removal_type,
    #     "n_posts": n_posts,
    #     "n_comments": n_comments,
    #     "n_output_rows": n_output_rows,
    # }
    # with open(output_file, 'a', encoding='utf-8') as target:
    #     target.write(json.dumps(summary, ensure_ascii=False) + '\n')

    print(f"Processed {n_input_lines} lines")
    print(f"Dropped {n_dropped_empty_or_removal_type} empty lines or removed/deleted posts/comments")
    print(f"Found {n_posts} posts and {n_comments} comments")
    print(f"Wrote {n_output_rows} rows to {output_file}")


def convert_input_file_to_list(input_file: str, tokenizer: AutoTokenizer, output_file: str) -> tuple[int, int, int, int, int, int]:
    """
    Takes in a JSONL file and extracts the text from each object.
    Writes to a json file with keys 'text', 'is_post', and 'token_length'.
    If the object has a 'body' field (is a comment), the text is written to the file with is_post set to False.
    If the object has a 'selftext' or 'title' field (is a post), the text is combined and written to the file with is_post set to True.
    """
    n_input_lines = n_dropped_empty_or_removal_type = n_posts = n_comments = n_output_rows = 0
    with open(input_file, 'r', encoding='utf-8') as source, open(output_file, 'w', encoding='utf-8') as target:
        for line in source:
            line = line.strip()
            if not line:
                continue
            n_input_lines += 1
            obj = json.loads(line)
            text, is_post = get_text_from_object(obj)
            if not text:
                n_dropped_empty_or_removal_type += 1
                continue
            if is_post:
                n_posts += 1
            else:
                n_comments += 1
            # NOTE: we will add chunking logic here later: for now, we are just filtering out long documents
            # if len(text.split()) > CHUNK_THRESHOLD_WORDS:
            #     n_skipped_long_docs += 1
            #     continue
            n_output_rows += 1
            target.write(json.dumps({
                'id': obj.get('id'),
                'text': text,
                'is_post': is_post,
                'token_length': calculate_token_length(tokenizer, text),
            }, ensure_ascii=False) + '\n')
    return n_input_lines, n_dropped_empty_or_removal_type, n_posts, n_comments, n_output_rows


def main():
    parser = argparse.ArgumentParser(description="Convert Reddit subthread data to suitable format for embeddings generation")
    parser.add_argument("-i", "--input_files", nargs='+', default=["data/r_chatGPT_posts.jsonl"], 
    help="Either provide posts, comments, or both file JSONL names. Takes at most two values.")
    parser.add_argument("-o", "--output_file", type=str, default="data/text_posts_for_embed.jsonl")

    args = parser.parse_args()
    if len(args.input_files) > 2:
        parser.error('--input_files accepts between 1 and 2 arguments only.')

    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    convert_input_files_to_json_object(args.input_files, tokenizer, args.output_file)
    print(f"Wrote texts to {args.output_file}")


if __name__ == '__main__':
    main()