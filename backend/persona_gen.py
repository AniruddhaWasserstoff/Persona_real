# backend/persona_gen.py

import os
import json
import logging
import re
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
        text = text.split("```", 2)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]

    # find balanced braces
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
                obj_str = text[start : i + 1]
                return json.loads(obj_str)
    raise ValueError("Unmatched braces in LLM output")


def generate_personas(
    clusters: Dict[int, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    For each cluster, generate a persona JSON object via ChatGroq.
    Ensures list fields use JSON arrays and normalizes set-like syntax.
    """
    personas: List[Dict[str, Any]] = []

    for label, members in clusters.items():
        if label == -1:
            continue

        # 1) System prompt
        system_msg = (
            "system",
            "You are an expert market researcher. "
            "Output EXACTLY one RAW JSON object containing the keys: "
            "persona_name (string), demographics (object), goals (object), "
            "pain_points (array of strings), channels (object), "
            "content_preferences (object), marketing_strategy (object). "
            "Do NOT include any other keys. Use JSON arrays [\"item1\",\"item2\"] for lists."
        )
        # 2) User prompt with cluster data
        human_msg = (
            "human",
            "Here are customer comments from one cluster—please derive a persona:\n" +
            "\n".join(p.get("text", "") for p in members)
        )

        # 3) Call LLM
        try:
            ai_msg = llm.invoke([system_msg, human_msg])
            raw = ai_msg.content

            # 4) Normalize set-like pain_points to JSON arrays
            raw = re.sub(
                r'("pain_points"\s*:\s*)\{\s*([^}]+?)\s*\}',
                lambda m: f"{m.group(1)}[{m.group(2).strip()}]",
                raw,
                flags=re.DOTALL
            )

            # 5) Extract JSON
            persona = _extract_json(raw)
            personas.append(persona)
            logger.info("Generated persona for cluster %d: %s", label, persona.get("persona_name"))
        except Exception as e:
            logger.error("Error generating persona for cluster %d: %s\nRAW OUTPUT:\n%s", label, e, raw if 'raw' in locals() else "")
            raise HTTPException(status_code=500, detail=f"Cluster {label}: {e}")

    return personas
