"""
Download r/chatGPT comments from the Arctic Shift API: https://arctic-shift.photon-reddit.com
"""
import sys
from datetime import datetime, timezone
 
import requests
from requests.adapters import HTTPAdapter, Retry
 
BASE = "https://arctic-shift.photon-reddit.com/api"
SUBREDDIT = "ChatGPT"
KEYWORD = "sycophan"

TEST_START = "2025-04-25"
TEST_END = "2025-05-02"
 
PAGE_LIMIT = 100
MAX_PAGES = 10_000

session = requests.Session()
session.mount(
    "https://",
    HTTPAdapter(max_retries=Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )),
)


def to_ts(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
 

def extract_items(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    raise ValueError(f"Unexpected response shape, inspect manually: {list(payload)[:200]!r}")


def fetch_all(params: dict) -> list[dict]:
    items: list[dict] = []
    seen_ids: set[str] = set()
    cursor = {**params, "sort": "asc", "limit": PAGE_LIMIT}

    for _ in range(MAX_PAGES):
        try:
            r = session.get(f"{BASE}/comments/search", params=cursor, timeout=60)
            r.raise_for_status()
            payload = r.json()
        except requests.exceptions.RequestException as e:
            print(f" WARNING: request failed for after={cursor.get('after')}: {e}")
            break
        
        err = str(payload["error"]) if isinstance(payload, dict) and "error" in payload else None
        if err:
            print(f"  WARNING: server returned an error for after={cursor.get('after')}: {err!r}; ")
            break
        
        batch = extract_items(payload)
        if not batch:
            break

        new = [c for c in batch if c["id"] not in seen_ids]
        items.extend(new)
        seen_ids.update(c["id"] for c in new)
        if not new:
            break
        if len(batch) < PAGE_LIMIT:
            break
        cursor["after"] = batch[-1]["created_utc"]
    else:
        raise RuntimeError(f"Hit MAX_PAGES={MAX_PAGES} without exhausting results")
    return items


def main() -> None:
    window = {
        "subreddit": SUBREDDIT,
        "after": to_ts(TEST_START),
        "before": to_ts(TEST_END),   
    }

    print(f"Downloading for ")

    # raw = fetch_all(window)
    # ground_truth = {c["id"] for c in raw if KEYWORD in c.get("body", "").lower()}
    # print(f"  {len(raw)} total comments, {len(ground_truth)} contain '{KEYWORD}' by substring")

    query = fetch_all({**window, "body": KEYWORD})
    print(f"  {len(query)} comments returned by query")


if __name__ == "__main__":
    main()