# backend/app.py

import os
import re
import json
import requests
import pandas as pd
import streamlit as st
from datetime import datetime
from requests.exceptions import RequestException

# ─── Config ───────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Business Toolkit", layout="wide")
st.sidebar.title("Tools")
page = st.sidebar.radio("Choose a tool", ["Existing Customer", "New Customer"])

# ─── Session State ────────────────────────────────────────────────────────────
_INITIAL_STATE = {
    "personas_df": None,
    "personas": [],
    "business_profile": None,
    "business_summary": "",
    "followup_questions": [],
    "competitor_videos": {},   # {competitor: [video dicts]}
    "video_comments": {},      # {video_id: [comment per question]}
    "comment_personas": [],
    "biz_defaults": {},        # autofill defaults from website extraction
}
for k, v in _INITIAL_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _api_post(path: str, payload: dict, timeout: int = 60) -> dict:
    """POST to backend with consistent error handling."""
    url = f"{API_BASE}/{path.lstrip('/')}"
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json() if r.content else {}
    except requests.exceptions.Timeout:
        st.error("🕒 Request timed out. Try again.")
        raise
    except RequestException as e:
        detail = None
        if getattr(e, "response", None) is not None:
            try:
                detail = e.response.json().get("detail")
            except Exception:
                detail = None
        st.error(f"API error: {e}{f' — {detail}' if detail else ''}")
        raise


def _reset_keys(keys):
    for k in keys:
        st.session_state[k] = _INITIAL_STATE[k]


def _valid_url(url: str) -> bool:
    return bool(re.match(r"^https?://", url.strip()))


def _year_or_default(raw, default: int) -> int:
    try:
        return int(raw)
    except Exception:
        return default


def _map_video_comments_to_questions(video_comments: dict, questions: list[str]) -> dict[str, list[str]]:
    """
    Convert {video_id: [c_for_q1, c_for_q2, ...]} to {question_text: [comments_across_videos]}.
    Ensures /comment_personas receives the expected shape.
    """
    qmap: dict[str, list[str]] = {q: [] for q in questions}
    for _vid, comments in (video_comments or {}).items():
        for i, q in enumerate(questions):
            if i < len(comments):
                c = (comments[i] or "").strip()
                if c:
                    qmap[q].append(c)
    # Drop empty lists to avoid empty personas
    return {q: cs for q, cs in qmap.items() if cs}


# ─── Existing Customer Flow ───────────────────────────────────────────────────
if page == "Existing Customer":
    st.title("Persona Builder (Existing Customer)")
    uploaded = st.file_uploader("Upload CSV (must include `customer_id`)", type="csv")

    col1, col2 = st.columns(2)
    with col1:
        generate_clicked = st.button("Generate Personas")
    with col2:
        if st.button("Reset Data"):
            _reset_keys(["personas_df", "personas"])
            st.rerun()

    if generate_clicked:
        if not uploaded:
            st.warning("Please upload a CSV file.")
            st.stop()

        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        if "customer_id" not in df.columns:
            st.error("CSV missing required column: `customer_id`.")
            st.stop()

        # Fill NaNs (strings -> "", numerics -> median)
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("")

        st.session_state.personas_df = df

        with st.spinner("Generating personas…"):
            data = _api_post("process_profiles", {"profiles": df.to_dict(orient="records")}, timeout=180)
            st.session_state.personas = data.get("personas", [])

    if st.session_state.personas_df is not None:
        st.subheader("Data Sample")
        st.dataframe(st.session_state.personas_df.head())

    if st.session_state.personas:
        st.success("Personas Generated")
        for persona in st.session_state.personas:
            st.json(persona)

# ─── New Customer Flow ────────────────────────────────────────────────────────
else:
    st.title("New Customer Onboarding")

    # Auto-fill from website (abstracted settings)
    auto = st.checkbox("🌐 Auto-fill from website")
    if auto:
        website_url = st.text_input("Website URL", placeholder="https://example.com")
        if st.button("Fetch data from site"):
            if not _valid_url(website_url):
                st.error("Please enter a valid URL starting with http:// or https://")
            else:
                with st.spinner("Fetching & analyzing site…"):
                    # NOTE: No max_pages / max_workers shown or sent; backend defaults apply.
                    st.session_state.biz_defaults = _api_post(
                        "extract_business_info",
                        {"website_url": website_url},
                        timeout=180,
                    )
                st.success("Auto-fill data loaded. You can tweak the fields below.")

    # Prepare defaults for the form
    raw = st.session_state.biz_defaults or {}
    ui_channel_options = ["Email", "Social Media", "Events", "SEO", "Partnerships", "Paid Ads"]

    defaults = {
        "name": raw.get("name", "") or "",
        "founded": raw.get("founded", "") or "",
        "locations": ", ".join(raw.get("locations", []) or []),
        "offerings": ", ".join(raw.get("offerings", []) or []),
        "price_range": raw.get("price_range", "") or "",
        "audience": ", ".join(raw.get("audience", []) or []),
        "usp": raw.get("usp", "") or "",
        "competitors": raw.get("competitors", []) or [],
        "channels": [opt for opt in ui_channel_options if opt.lower() in [c.lower() for c in (raw.get("channels", []) or [])]],
        "goals": "; ".join(raw.get("goals", []) or []),
    }

    # Business profile form
    with st.form("biz_form"):
        name = st.text_input("Business Name", value=defaults["name"])
        current_year = datetime.now().year
        default_founded = _year_or_default(defaults["founded"], min(current_year, 2025))
        founded = st.number_input(
            "Year Founded",
            min_value=1900,
            max_value=max(current_year, 2025),
            value=default_founded,
        )

        locations = st.text_input("Location(s)", value=defaults["locations"], help="e.g. Mumbai; Pune")
        offerings = st.text_area("Products / Services (comma-separated)", value=defaults["offerings"])
        price_range = st.text_input("Price Range (e.g. ₹200–₹800)", value=defaults["price_range"])
        audience = st.text_area("Ideal Customers (demographics, region)", value=defaults["audience"])
        usp = st.text_area("Unique Selling Proposition", value=defaults["usp"])

        # --- Competitors input UX ---
        competitors_suggested = defaults.get("competitors", []) or []
        custom_comp_text = ""
        selected_from_suggestions: list[str] = []

        if competitors_suggested:
            selected_from_suggestions = st.multiselect(
                "Key Competitors (select from suggestions)",
                options=competitors_suggested,
                default=[]
            )
            custom_comp_text = st.text_input(
                "Add more competitors (comma-separated)",
                placeholder="e.g., Curry House, Tandoori Express, Urban Masala",
                help="Type any additional competitor names and hit Enter."
            )
        else:
            custom_comp_text = st.text_area(
                "Key Competitors (comma-separated)",
                placeholder="e.g., Curry House, Tandoori Express, Urban Masala",
                help="No suggestions available — type competitors separated by commas."
            )

        typed_custom = [c.strip() for c in (custom_comp_text or "").split(",") if c.strip()]
        competitors = list(dict.fromkeys([*selected_from_suggestions, *typed_custom]))  # unique, keep order

        channels = st.multiselect("Marketing Channels", ui_channel_options, default=defaults["channels"])
        goals = st.text_area("Top 3 Goals (semicolon-separated)", value=defaults["goals"])
        submitted = st.form_submit_button("Generate Business Profile")

    # Reset all state
    if st.button("Reset Data"):
        _reset_keys(list(_INITIAL_STATE.keys()))
        st.rerun()

    # On submit: run the entire pipeline automatically
    if submitted:
        goals_list = [g.strip() for g in goals.split(";") if g.strip()]
        biz_payload = {
            "name": name.strip(),
            "founded": str(int(founded)),
            "locations": locations.strip(),
            "offerings": offerings.strip(),
            "price_range": price_range.strip(),
            "audience": audience.strip(),
            "usp": usp.strip(),
            "competitors": competitors,
            "channels": channels,
            "goals": goals_list,
        }

        # Step 1: Summarize business (structured profile)
        with st.spinner("Creating structured business profile…"):
            st.session_state.business_profile = _api_post("summarize_business", {"business": biz_payload}, timeout=120)

        # Step 2: Human-friendly summary
        with st.spinner("Generating human-friendly summary…"):
            summary_resp = _api_post("summarize_profile", st.session_state.business_profile, timeout=90)
            st.session_state.business_summary = summary_resp.get("summary", "").strip()

        # Step 3+: Competitor pipeline (only if competitors present)
        comps = (st.session_state.business_profile or {}).get("competitors", []) or competitors
        if comps:
            # 3.1 Generate follow-up questions
            with st.spinner("Generating competitor follow-up questions…"):
                q_resp = _api_post(
                    "generate_followup_queries",
                    {"summary": st.session_state.business_summary, "topic": "competitors", "competitors": comps},
                    timeout=60,
                )
                st.session_state.followup_questions = q_resp.get("questions", [])[:3]

            # 3.2 Fetch competitor videos
            vids_map: dict[str, list[dict]] = {}
            with st.spinner("Searching top YouTube videos for competitors…"):
                for comp in comps:
                    try:
                        v_resp = _api_post("youtube_search", {"query": comp, "order": "viewCount", "max_results": 5}, timeout=90)
                        vids_map[comp] = v_resp.get("videos", [])
                    except Exception:
                        # already surfaced by _api_post; continue with others
                        pass
            st.session_state.competitor_videos = vids_map

            # 3.3 Fetch per-question comments for each video
            all_ids = [v["id"] for vids in st.session_state.competitor_videos.values() for v in vids if v.get("id")]
            if all_ids and st.session_state.followup_questions:
                with st.spinner("Fetching & ranking top comments per question…"):
                    st.session_state.video_comments = _api_post(
                        "youtube_comments_filtered",
                        {"video_ids": all_ids, "questions": st.session_state.followup_questions},
                        timeout=180,
                    )

                # 3.4 Generate personas from mapped comments
                qmap = _map_video_comments_to_questions(st.session_state.video_comments, st.session_state.followup_questions)
                if qmap:
                    with st.spinner("Generating personas from competitor insights…"):
                        r_p = _api_post("comment_personas", qmap, timeout=120)
                        st.session_state.comment_personas = r_p.get("personas", [])
        else:
            st.info("No competitors provided — skipping competitor insights pipeline.")

    # ─── Display Results (no extra clicks) ─────────────────────────────────────
    if st.session_state.business_summary:
        st.subheader("Profile Summary")
        st.write(st.session_state.business_summary)

        if st.session_state.followup_questions:
            st.markdown("**Competitor Follow-Up Questions:**")
            for i, q in enumerate(st.session_state.followup_questions, 1):
                st.write(f"{i}. {q}")

        if st.session_state.competitor_videos:
            st.subheader("Competitor Videos (Top 5 by Views)")
            for comp, vids in st.session_state.competitor_videos.items():
                st.markdown(f"**{comp}**")
                for v in vids:
                    title = v.get("title", "Untitled")
                    url = v.get("url", "")
                    views = v.get("viewCount", 0)
                    st.write(f"- [{title}]({url}) — {views:,} views")

        if st.session_state.video_comments:
            st.subheader("Top Semantically Relevant Comments")
            for vid, comms in st.session_state.video_comments.items():
                st.write(f"**Video {vid}:**")
                for c in comms:
                    st.write(f"- {c}")

        if st.session_state.comment_personas:
            st.subheader("Customer Personas from Competitor Insights")
            for persona in st.session_state.comment_personas:
                st.json(persona)
