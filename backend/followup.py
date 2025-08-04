# backend/followup.py

import json
import re
import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, List
from .business import _call_groq_with_retries, SUMMARY_MODEL

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/generate_followup_queries", response_model=Dict[str, List[str]])
async def generate_followup_queries(request: Dict[str, str]) -> Dict[str, List[str]]:
    summary = request.get("summary")
    if not summary:
        raise HTTPException(status_code=400, detail="Missing 'summary'")

    # 1) Prompt the LLM for exactly three JSON questions
    system_msg = (
        "You are a business consultant. "
        "Output EXACTLY 3 follow-up questions as a JSON array of strings, "
        "with no extra text or numbering."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": summary}
    ]
    payload = {"model": SUMMARY_MODEL, "messages": messages, "max_tokens": 150}

    data = _call_groq_with_retries(payload)
    raw = data["choices"][0]["message"]["content"]
    logger.debug("LLM raw output for follow-ups: %s", raw)

    # 2) Try direct parse
    try:
        questions = json.loads(raw.strip())
    except json.JSONDecodeError:
        # 3) Fallback: extract the first [...] block
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not m:
            logger.error("No JSON array found in LLM output: %s", raw)
            raise HTTPException(500, f"No JSON array found in LLM output:\n{raw}")
        try:
            questions = json.loads(m.group())
        except json.JSONDecodeError:
            logger.error("Failed to parse extracted JSON array: %s", m.group())
            raise HTTPException(500, f"Could not parse JSON array from LLM output:\n{raw}")

    # 4) Validate
    if (
        not isinstance(questions, list)
        or len(questions) < 3
        or not all(isinstance(q, str) for q in questions)
    ):
        logger.error("Invalid questions format: %r", questions)
        raise HTTPException(500, f"Expected list of >=3 strings, got: {questions}")

    # 5) Return exactly three
    return {"questions": questions[:3]}
