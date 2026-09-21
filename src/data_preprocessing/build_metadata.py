"""Build metadata parquet files for datasets.
Stores: 1) meta_comments.parquet and 2) meta_posts.parquet.

The text files used for embedding (text_*_for_embed_*.jsonl) only carry {id, text, is_post, token_length};
everything needed for analysis (thread structure, timestamps, engagement) lives here, keyed by id.

Comments fields: id, link_id, parent_id, author, created_utc, score, controversiality, permalink
Posts fields:    id, author, created_utc, score, num_comments, upvote_ratio, title, permalink,
                 selftext_status (deleted / removed / empty / text),
                 post_kind (text / image / video / link, derived from url),
                 link_flair_text

Reads the raw dumps in streaming fashion (constant memory) and skips all other columns.

Usage:
    Posts only: python build_metadata.py --posts_only
    Comments only: python build_metadata.py --comments_only
    Both: python build_metadata.py
"""

import argparse
import time
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.json as paj
import pyarrow.parquet as pq

CHUNK_SIZE = 64  # MB read per block

COMMENTS_SCHEMA = pa.schema([
    ('id', pa.string()),
    ('link_id', pa.string()),
    ('parent_id', pa.string()),
    ('author', pa.string()),
    ('created_utc', pa.int64()),
    ('score', pa.int64()),
    ('controversiality', pa.int64()),
    ('permalink', pa.string()),
])

# selftext and url are only read to derive selftext_status / post_kind; they are replaced before writing
POSTS_SCHEMA = pa.schema([
    ('id', pa.string()),
    ('author', pa.string()),
    ('created_utc', pa.int64()),
    ('score', pa.int64()),
    ('num_comments', pa.int64()),
    ('upvote_ratio', pa.float64()),
    ('title', pa.string()),
    ('permalink', pa.string()),
    ('selftext', pa.string()),
    ('url', pa.string()),
    ('link_flair_text', pa.string()),
])

IMAGE_HOST_RE = r'^https?://(i\.redd\.it|preview\.redd\.it|i\.imgur\.com|imgur\.com|www\.reddit\.com/gallery)/'
VIDEO_HOST_RE = r'^https?://(v\.redd\.it|youtu\.be|www\.youtube\.com)/'
SELF_POST_RE = r'^https?://(www\.|old\.)?reddit\.com/r/[^/]+/comments/'  # a self post's url is its own permalink


def _derive_post_columns(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Replace raw selftext/url with compact derived columns so the parquet stays small."""
    selftext = pc.utf8_trim_whitespace(pc.fill_null(batch.column('selftext'), ''))
    selftext_status = pc.if_else(pc.equal(selftext, '[deleted]'), 'deleted',
                      pc.if_else(pc.equal(selftext, '[removed]'), 'removed',
                      pc.if_else(pc.equal(selftext, ''), 'empty', 'text')))

    url = pc.fill_null(batch.column('url'), '')
    post_kind = pc.if_else(pc.match_substring_regex(url, IMAGE_HOST_RE), 'image',
                pc.if_else(pc.match_substring_regex(url, VIDEO_HOST_RE), 'video',
                pc.if_else(pc.match_substring_regex(url, SELF_POST_RE), 'text', 'link')))

    flair_idx = batch.schema.get_field_index('link_flair_text')
    flair = pc.utf8_trim_whitespace(batch.column('link_flair_text'))  # raw flairs carry trailing spaces

    batch = batch.set_column(batch.schema.get_field_index('selftext'), 'selftext_status', selftext_status)
    batch = batch.set_column(batch.schema.get_field_index('url'), 'post_kind', post_kind)
    batch = batch.set_column(flair_idx, 'link_flair_text', flair)
    return batch


def build_metadata(input_file: str, output_file: str, schema: pa.Schema, is_posts: bool = False) -> None:
    """Stream a raw JSONL dump into a parquet file containing only the columns in `schema`.
    Note: pyarrow's JSON reader is quicker than pandas (C++ backend, unparsed columns are skipped entirely),
    and open_json + ParquetWriter keeps memory flat regardless of dump size (the comments dump is 14 GB)."""
    read_options = paj.ReadOptions(block_size=CHUNK_SIZE * 1024 * 1024)
    parse_options = paj.ParseOptions(
        explicit_schema=schema,
        unexpected_field_behavior='ignore',
    )

    t0 = time.time()
    n_rows = 0
    writer = None
    with paj.open_json(input_file, read_options=read_options, parse_options=parse_options) as reader:
        for batch in reader:
            if is_posts:
                batch = _derive_post_columns(batch)
            if writer is None:
                writer = pq.ParquetWriter(output_file, batch.schema, compression='snappy')
            writer.write_batch(batch)
            n_rows += batch.num_rows
    if writer is not None:
        writer.close()
    print(f"Wrote {n_rows} rows -> {output_file} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build metadata parquet sidecars from the raw Reddit dumps")
    parser.add_argument("--posts_only", action='store_true')
    parser.add_argument("--comments_only", action='store_true')
    parser.add_argument("--input_file_comments", type=str, default='data/r_chatGPT_comments_2022-12-22_2026-9-18.jsonl')
    parser.add_argument("--output_file_comments", type=str, default='data/meta_comments.parquet')
    parser.add_argument("--input_file_posts", type=str, default='data/r_chatGPT_posts_2022-12-22_2026-9-18.jsonl')
    parser.add_argument("--output_file_posts", type=str, default='data/meta_posts.parquet')
    args = parser.parse_args()

    if not args.posts_only:
        build_metadata(args.input_file_comments, args.output_file_comments, COMMENTS_SCHEMA)
    if not args.comments_only:
        build_metadata(args.input_file_posts, args.output_file_posts, POSTS_SCHEMA, is_posts=True)
