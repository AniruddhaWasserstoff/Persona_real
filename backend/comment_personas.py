# backend/comment_personas.py

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any

from .embeddings import upsert_embeddings
from .clustering import cluster_embeddings
from .persona_gen import generate_personas

router = APIRouter()

@router.post("/comment_personas", response_model=Dict[str, Any])
async def comment_personas(request: Dict[str, List[str]]) -> Dict[str, Any]:
    # Build profile objects with integer customer_id for Qdrant
    profiles = []
    idx = 0
    for question, comments in request.items():
        for c in comments:
            profiles.append({
                "customer_id": idx,          # now an integer
                "text": c,
                "source_question": question
            })
            idx += 1

    if not profiles:
        raise HTTPException(status_code=400, detail="No comments provided")

    # Embed, cluster, and generate personas
    triples = upsert_embeddings(profiles)
    clusters = cluster_embeddings(triples)
    personas = generate_personas(clusters)
    return {"personas": personas}
