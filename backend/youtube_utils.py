# backend/youtube_utils.py

import os
import logging
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import List
from langdetect import detect
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

# 1. Read API key from .env
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    raise RuntimeError("YOUTUBE_API_KEY missing in .env")

# 2. Initialize the YouTube Data API v3 client
yt = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# 3. Load the embedding model once for semantic ranking
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def search_top_video_ids(query: str, max_results: int = 5) -> List[str]:
    """
    Return up to `max_results` video IDs for a search query.
    """
    try:
        resp = yt.search().list(
            part="id",
            q=query,
            type="video",
            maxResults=max_results
        ).execute()
        return [item["id"]["videoId"] for item in resp.get("items", [])]
    except HttpError as e:
        logger.warning("YouTube search failed for query '%s': %s", query, e)
        return []


def fetch_top_comments(
    video_id: str,
    query: str,
    max_comments: int = 3,
    pool_size: int = 50,
    min_similarity: float = 0.3
) -> List[str]:
    """
    Fetch up to `pool_size` comments by relevance, filter to English, and return the top
    `max_comments` comments most semantically similar to `query`.

    - Skips videos with comments disabled.
    - `min_similarity` thresholds similarity below which comments are discarded.
    """
    # 1) Pull a pool of comments, handle disabled comments
    try:
        resp = yt.commentThreads().list(
            part="snippet",
            videoId=video_id,
            order="relevance",
            maxResults=min(pool_size, 100)
        ).execute()
    except HttpError as e:
        # 403 for commentsDisabled
        if hasattr(e, 'resp') and e.resp.status == 403:
            logger.info("Comments disabled for video %s, skipping.", video_id)
            return []
        logger.warning("YouTube commentThreads failed for video %s: %s", video_id, e)
        return []

    # Extract the raw comment text
    raw_comments = [
        item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        for item in resp.get("items", [])
    ]

    # 2) Filter to English only
    comments = []
    for c in raw_comments:
        try:
            if detect(c) == "en":
                comments.append(c)
        except Exception:
            continue

    if not comments:
        return []

    # 3) Embed query and comments
    q_emb = embedder.encode(query, convert_to_tensor=True)
    c_embs = embedder.encode(comments, convert_to_tensor=True)

    # 4) Compute cosine similarities
    sims = util.cos_sim(q_emb, c_embs)[0]

    # 5) Rank and select top comments above threshold
    sorted_indices = sims.argsort(descending=True)
    top_comments: List[str] = []
    for idx in sorted_indices:
        if float(sims[idx]) < min_similarity:
            break
        top_comments.append(comments[int(idx)])
        if len(top_comments) >= max_comments:
            break

    return top_comments
