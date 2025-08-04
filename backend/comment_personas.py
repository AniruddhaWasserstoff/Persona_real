# backend/comment_personas.py

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any

from .embeddings import upsert_embeddings
from .persona_gen import generate_personas

router = APIRouter()

@router.post("/comment_personas", response_model=Dict[str, Any])
async def comment_personas(request: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Build personas directly per follow-up question by grouping comments,
    bypassing HDBSCAN so each question yields one persona.
    """
    # 1) Flatten comments into profiles
    profiles: List[Dict[str, Any]] = []
    idx = 0
    for question, comments in request.items():
        for c in comments:
            profiles.append({
                "customer_id": idx,
                "text": c,
                "source_question": question
            })
            idx += 1

    if not profiles:
        raise HTTPException(status_code=400, detail="No comments provided")

    # 2) Embed all profiles
    triples = upsert_embeddings(profiles)

    # 3) Group by source_question instead of clustering
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for t in triples:
        q = t["payload"]["source_question"]
        grouped.setdefault(q, []).append(t["payload"])

    # 4) Re-index for generate_personas
    clusters: Dict[int, List[Dict[str, Any]]] = {
        i: members for i, members in enumerate(grouped.values())
    }

    # 5) Generate one persona per question
    personas = generate_personas(clusters)
    return {"personas": personas}
