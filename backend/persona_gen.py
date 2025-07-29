# backend/persona_gen.py

import os
import json
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv
from fastapi import HTTPException
from langchain_groq import ChatGroq

# ─── Load config & setup logging ─────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in the environment")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# ─── Instantiate Groq LLM ────────────────────────────────────────────────────
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_tokens=600,
    max_retries=2,
    reasoning_format=None,
    timeout=None
)

def _extract_json(raw: str) -> Dict[str, Any]:
    """
    Pull the first JSON object out of the LLM’s raw string.
    """
    text = raw.strip()
    # strip markdown fences
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        text = "\n".join(lines)
    if text.endswith("```"):
        lines = text.splitlines()[:-1]
        text = "\n".join(lines)

    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in LLM output")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("Unmatched braces in LLM output")

def generate_personas(
    clusters: Dict[int, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    For each non-noise cluster, ask Groq’s Llama model (via ChatGroq)
    to emit exactly one raw JSON persona object.
    """
    personas: List[Dict[str, Any]] = []

    for label, members in clusters.items():
        if label == -1:
            continue

        # build the messages in the (role, content) tuple format
        system_msg = (
            "system",
            "You are a marketing strategist. OUTPUT MUST BE ONE RAW JSON OBJECT—"
            "no markdown, no explanation, keys and string values in double quotes."
        )
        human_msg = (
            "human",
            "Cluster profiles:\n"
            f"{json.dumps(members, ensure_ascii=False)}\n\n"
            "1) Assign a unique persona_name.\n"
            "2) Describe demographics, goals, pain_points, channels, content_preferences.\n"
            "3) Recommend marketing_strategy with keys awareness, consideration, decision.\n\n"
            "Return exactly one JSON object with keys: persona_name, demographics, goals, "
            "pain_points, channels, content_preferences, marketing_strategy."
        )

        try:
            ai_msg = llm.invoke([system_msg, human_msg])
            raw = ai_msg.content
            persona = _extract_json(raw)
            personas.append(persona)
            logger.info("Generated persona for cluster %d: %s", label, persona.get("persona_name"))
        except Exception as e:
            logger.error("Error generating persona for cluster %d: %s\nRAW OUTPUT:\n%s", label, e, raw if 'raw' in locals() else "")
            raise HTTPException(status_code=500, detail=f"Cluster {label}: {e}")

    return personas
