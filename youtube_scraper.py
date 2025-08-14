# youtube_scraper_terminal.py

import os
import json
import logging
from dotenv import load_dotenv
from langdetect import detect
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sentence_transformers import SentenceTransformer, util

# -------------------------
# Setup logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("youtube_scraper")

# -------------------------
# Load API key
# -------------------------
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    raise RuntimeError("YOUTUBE_API_KEY not found in .env")

yt = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------
# Functions
# -------------------------
def search_top_video_ids(query, max_results=5):
    try:
        resp = yt.search().list(
            part="id",
            q=query,
            type="video",
            maxResults=max_results
        ).execute()
        return [item["id"]["videoId"] for item in resp.get("items", [])]
    except HttpError as e:
        logger.warning("YouTube search failed: %s", e)
        return []


def fetch_video_metadata(video_ids):
    if not video_ids:
        return []
    try:
        resp = yt.videos().list(
            part="snippet,statistics",
            id=",".join(video_ids),
            maxResults=len(video_ids)
        ).execute()
    except HttpError as e:
        logger.error("YouTube videos().list failed: %s", e)
        return []

    videos = []
    for item in resp.get("items", []):
        videos.append({
            "id": item["id"],
            "title": item["snippet"]["title"],
            "url": f"https://youtu.be/{item['id']}",
            "viewCount": int(item.get("statistics", {}).get("viewCount", 0))
        })
    videos.sort(key=lambda v: v["viewCount"], reverse=True)
    return videos


def fetch_top_comments(video_id, question, max_comments=3, pool_size=50, min_similarity=0.3):
    try:
        resp = yt.commentThreads().list(
            part="snippet",
            videoId=video_id,
            order="relevance",
            maxResults=min(pool_size, 100),
            textFormat="plainText"
        ).execute()
    except HttpError as e:
        if hasattr(e, "resp") and getattr(e.resp, "status", None) == 403:
            logger.info("Comments disabled for video %s, skipping.", video_id)
            return []
        logger.warning("commentThreads failed for %s: %s", video_id, e)
        return []

    raw_comments = [
        item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        for item in resp.get("items", [])
    ]

    comments = []
    for c in raw_comments:
        try:
            if detect(c) == "en":
                comments.append(c)
        except Exception:
            continue

    if not comments:
        return []

    q_emb = embedder.encode(question, convert_to_tensor=True)
    c_embs = embedder.encode(comments, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, c_embs)[0]
    sorted_indices = sims.argsort(descending=True)

    top_comments = []
    for idx in sorted_indices:
        if float(sims[idx]) < min_similarity:
            break
        top_comments.append(comments[int(idx)])
        if len(top_comments) >= max_comments:
            break

    return top_comments


# -------------------------
# Main Interactive Run
# -------------------------
if __name__ == "__main__":
    print("📹 YouTube Scraper - Interactive Mode 📹\n")

    query = input("🔍 Enter search query: ").strip()
    question = input("💬 Enter your question for filtering comments: ").strip()
    max_results = int(input("🎯 Number of videos to fetch: ").strip() or 5)
    max_comments = int(input("📝 Number of top comments per video: ").strip() or 3)

    print("\n🔎 Searching YouTube...")
    ids = search_top_video_ids(query, max_results)
    videos = fetch_video_metadata(ids)

    results = []
    for v in videos:
        print(f"\n▶ {v['title']} ({v['url']})")
        comments = fetch_top_comments(v["id"], question, max_comments=max_comments)
        for idx, c in enumerate(comments, start=1):
            print(f"   {idx}. {c}")
        results.append({"video": v, "top_comments": comments})

    print("\n✅ Done! Results in JSON format below:\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))
