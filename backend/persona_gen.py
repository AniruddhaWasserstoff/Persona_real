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
    Extract the first JSON object from LLM output, with fallback quoting for numeric ranges.
    """
    text = raw.strip()
    # Remove code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
    if text.endswith("```"):
        text = text.rsplit("```")[0]

    # Find balanced braces
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
                try:
                    return json.loads(obj_str)
                except json.JSONDecodeError:
                    # Fallback: quote simple numeric ranges
                    fixed = re.sub(r'("\w+"\s*:\s*)(\d+-\d+)', r'\1"\2"', obj_str)
                    return json.loads(fixed)
    raise ValueError("Unmatched braces in LLM output")


def generate_personas(
    clusters: Dict[int, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Generate a distinct persona JSON for each cluster using ChatGroq.
    Instructs the LLM to tailor each persona to its cluster comments and avoid repetition across clusters.
    """
    personas: List[Dict[str, Any]] = []
    generated_names: List[str] = []

    for label, members in clusters.items():
        if label == -1:
            continue

        # List of previous persona names for uniqueness constraint
        prev_list = ", ".join(generated_names) if generated_names else "none"

        # 1) System prompt: enforce valid JSON and distinctness
        system_msg = (
            "system",
            (
                f"You are an expert market researcher. Create exactly one JSON persona for cluster {label}. "
                f"Previously generated persona names: {prev_list}. "
                "Ensure this persona is distinct in persona_name, demographics, goals, and pain_points from all previous ones. "
                "Include ONLY these keys: persona_name (string), demographics (object), goals (object), "
                "pain_points (array of strings), channels (object), content_preferences (object), marketing_strategy (object). "
                "All keys and values must be double-quoted. Numeric ranges (e.g. \"28-35\") must be strings. "
                "Do NOT include comments, markdown, or trailing commas. Use JSON arrays for lists."
            )
        )

        # 2) Human prompt: provide cluster-specific comments
        texts = [p.get("text", "") for p in members if p.get("text")]
        human_msg = (
            "human",
            (
                f"Here are the comments for cluster {label}. Derive a unique persona JSON object:\n\n" +
                "\n".join(texts)
            )
        )

        # 3) Invoke the LLM
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

            # 5) Extract JSON with fallback
            persona = _extract_json(raw)
            personas.append(persona)
            # Track name for next iteration
            name = persona.get("persona_name", "").strip()
            if name:
                generated_names.append(name)

            logger.info("Cluster %d persona generated: %s", label, name)
        except Exception as e:
            logger.error(
                "Error generating persona for cluster %d: %s\nRaw output:\n%s",
                label, e, raw if 'raw' in locals() else ""
            )
            raise HTTPException(status_code=500, detail=f"Cluster {label}: {e}")

    return personas
