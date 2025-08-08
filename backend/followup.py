# backend/followup.py

import json
import re
import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from .business import _call_groq_with_retries, SUMMARY_MODEL

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post(
    "/generate_followup_queries",
    response_model=Dict[str, List[str]]
)
async def generate_followup_queries(request: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Generate exactly 3 follow-up questions based on the business summary.
    If topic == 'competitors' and a competitors list is provided,
    the questions will reference competitors by name.
    """
    summary     = request.get("summary")
    topic       = request.get("topic")
    competitors = request.get("competitors", [])

    if not summary:
        raise HTTPException(status_code=400, detail="Missing 'summary'")

    # 1) Construct the system prompt
    if topic == "competitors" and competitors:
        comp_str = ", ".join(competitors)
        system_msg = (
            "You are a business consultant. "
            "Given the business summary and these competitors, "
            "output EXACTLY 3 follow-up questions as a JSON array of strings, "
            "focused on each competitor's impact and strategy, referencing competitors by name.\n\n"
            "Business Summary:\n"
            f"{summary}\n\n"
            f"Key Competitors: {comp_str}"
        )
    else:
        system_msg = (
            "You are a business consultant. "
            "Output EXACTLY 3 follow-up questions as a JSON array of strings, "
            "with no extra text or numbering, based on this summary:\n\n"
            f"{summary}"
        )

    messages = [{"role": "system", "content": system_msg}]
    payload = {"model": SUMMARY_MODEL, "messages": messages, "max_tokens": 150}

    # 2) Call the LLM
    data = _call_groq_with_retries(payload)
    raw = data["choices"][0]["message"]["content"]
    logger.debug("LLM raw output for follow-ups: %s", raw)

    # 3) Parse the JSON array of questions
    try:
        questions = json.loads(raw.strip())
    except json.JSONDecodeError:
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not match:
            logger.error("No JSON array found in LLM output: %s", raw)
            raise HTTPException(
                status_code=500,
                detail=f"No JSON array found in LLM output:\n{raw}"
            )
        try:
            questions = json.loads(match.group())
        except json.JSONDecodeError:
            logger.error("Failed to parse extracted JSON array: %s", match.group())
            raise HTTPException(
                status_code=500,
                detail=f"Could not parse JSON array from LLM output:\n{raw}"
            )

    # 4) Validate format
    if (
        not isinstance(questions, list)
        or len(questions) < 3
        or not all(isinstance(q, str) for q in questions)
    ):
        logger.error("Invalid questions format: %r", questions)
        raise HTTPException(
            status_code=500,
            detail=f"Expected list of >=3 strings, got: {questions}"
        )

    # 5) Return exactly three questions
    return {"questions": questions[:3]}
