# backend/business.py

import os
import time
import random
import json
import requests
from dotenv import load_dotenv
from fastapi import HTTPException

# ─── Load environment ──────────────────────────────────────────────────────────
load_dotenv()
API_KEY        = os.getenv("GROQ_API_KEY")
ENDPOINT       = "https://api.groq.com/openai/v1/chat/completions"
BUSINESS_MODEL = "llama-3.1-8b-instant"  # for JSON profile
SUMMARY_MODEL  = "llama-3.1-8b-instant"  # for human-readable summary

if not API_KEY:
    raise ValueError("GROQ_API_KEY must be set in the environment")

# ─── Helper: call Groq with retry/backoff ──────────────────────────────────────
def _call_groq_with_retries(payload: dict, max_retries: int = 5, backoff: float = 1.0) -> dict:
    for attempt in range(max_retries):
        resp = requests.post(
            ENDPOINT,
            json=payload,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            wait = backoff * (2 ** attempt) + random.random()
            time.sleep(wait)
            continue
        resp.raise_for_status()
    raise HTTPException(500, f"Rate limit: exceeded {max_retries} retries")

# ─── Helper: extract the first {...} JSON block from LLM output ───────────────
def _extract_json(raw: str) -> dict:
    txt = raw.strip()
    # strip markdown fences if present
    if txt.startswith("```"):
        txt = "\n".join(txt.split("\n")[1:])
    if txt.endswith("```"):
        txt = "\n".join(txt.split("\n")[:-1])

    start = txt.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM output")

    depth = 0
    for i, ch in enumerate(txt[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = txt[start : i + 1]
                return json.loads(snippet)
    raise ValueError("Unmatched braces in LLM output")

# ─── Main: build structured Business Profile JSON ─────────────────────────────
def summarize_business(biz: dict) -> dict:
    """
    Input:  biz = {
      name, founded, locations, offerings, price_range,
      audience, usp, competitors, channels, goals, voice
    }
    Output: a JSON dict with exactly those keys (values as strings or lists).
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a business analyst. "
                "Output MUST be raw JSON only, with no explanations. "
                "Produce exactly one JSON object with keys: "
                "name, founded, locations, offerings, price_range, "
                "audience, usp, competitors, channels, goals, voice. "
                "All values must be strings or lists of strings."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(biz, ensure_ascii=False),
        },
    ]
    payload = {
        "model": BUSINESS_MODEL,
        "messages": messages,
        "max_tokens": 512,
    }
    data = _call_groq_with_retries(payload)
    raw = data["choices"][0]["message"]["content"]
    try:
        return _extract_json(raw)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse business profile JSON: {e}\nRaw output:\n{raw}",
        )

# ─── Secondary: turn structured profile into a paragraph ──────────────────────
def summarize_profile(profile: dict) -> str:
    """
    Input:  profile = the dict returned by summarize_business()
    Output: A concise, human-friendly paragraph preserving all key details.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a business analyst. "
                "Summarize the following BUSINESS PROFILE JSON into one concise paragraph, "
                "preserving all key details and keywords."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(profile, ensure_ascii=False),
        },
    ]
    payload = {
        "model": SUMMARY_MODEL,
        "messages": messages,
        "max_tokens": 200,
    }
    data = _call_groq_with_retries(payload)
    return data["choices"][0]["message"]["content"].strip()
