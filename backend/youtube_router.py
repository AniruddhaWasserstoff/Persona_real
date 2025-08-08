# backend/youtube_router.py

import re
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from .youtube_utils import yt, search_top_video_ids, fetch_top_comments

router = APIRouter()

@router.post("/youtube_search", response_model=Dict[str, List[Dict[str, Any]]])
async def youtube_search(request: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search YouTube for videos matching `query`, ordered by `order`,
    returning up to `max_results` videos with title, URL, and viewCount.
    """
    query = request.get("query")
    order = request.get("order", "viewCount")
    max_results = request.get("max_results", 5)

    if not query:
        raise HTTPException(status_code=400, detail="Missing required field: 'query'")

    # 1) Retrieve top video IDs for the search query
    video_ids = search_top_video_ids(query, max_results)

    videos: List[Dict[str, Any]] = []
    try:
        # 2) Fetch detailed metadata for each video ID
        resp = yt.videos().list(
            part="snippet,statistics",
            id=",".join(video_ids),
            maxResults=len(video_ids)
        ).execute()

        for item in resp.get("items", []):
            videos.append({
                "id": item["id"],
                "title": item["snippet"]["title"],
                "url": f"https://youtu.be/{item['id']}",
                "viewCount": int(item["statistics"].get("viewCount", 0))
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YouTube API error: {e}")

    return {"videos": videos}


@router.post("/youtube_comments_filtered", response_model=Dict[str, List[str]])
async def youtube_comments_filtered(request: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    For a list of video IDs and follow-up questions, fetch top semantically
elevant comments per video-question pair, returning one best comment per video.
    """
    video_ids = request.get("video_ids")
    questions = request.get("questions")

    if not video_ids or not isinstance(video_ids, list):
        raise HTTPException(status_code=400, detail="Missing or invalid 'video_ids'")
    if not questions or not isinstance(questions, list):
        raise HTTPException(status_code=400, detail="Missing or invalid 'questions'")

    output: Dict[str, List[str]] = {}

    for vid in video_ids:
        best_comments: List[str] = []
        for q in questions:
            # fetch_top_comments returns comments ranked by semantic relevance
            try:
                comments = fetch_top_comments(vid, q)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error fetching comments for video {vid}: {e}")
            if comments:
                best_comments.append(comments[0])
        # map video ID to its top comments list
        output[vid] = best_comments

    return output
