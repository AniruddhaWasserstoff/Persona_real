# backend/youtube_router.py

import re
from fastapi import APIRouter, HTTPException
from typing import Dict, List
from .youtube_utils import search_top_video_ids, fetch_top_comments

router = APIRouter()

def simplify_query(q: str) -> str:
    """
    Simplify a follow-up question by stripping out brand names (capitalized words)
    and numeric tokens so the search focuses on core intent.
    """
    # Remove capitalized words (e.g., brand names) and numbers/percent signs
    simplified = re.sub(r"\b[A-Z][a-zA-Z]+\b|\d+%?", "", q)
    # Collapse multiple spaces to one
    simplified = re.sub(r"\s+", " ", simplified).strip()
    return simplified or q

@router.post("/youtube_comments", response_model=Dict[str, List[str]])
async def youtube_comments(request: Dict[str, List[str]]) -> Dict[str, List[str]]:
    questions = request.get("questions")
    if not questions or len(questions) != 3:
        raise HTTPException(status_code=400, detail="Send exactly 3 questions")

    output: Dict[str, List[str]] = {}
    for q in questions:
        # 1) Simplify the question for searching
        term = simplify_query(q)
        # 2) Get top video IDs for that term
        vids = search_top_video_ids(term)
        collected: List[str] = []

        for vid in vids:
            # 3) Fetch semantically ranked comments for the original question
            comments = fetch_top_comments(vid, q)
            collected.extend(comments)
            if len(collected) >= 3:
                break

        # 4) Keep only the first 3 comments
        output[q] = collected[:3]

    return output
