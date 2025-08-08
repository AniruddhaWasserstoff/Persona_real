import streamlit as st
import pandas as pd
import requests
import os

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
st.set_page_config(page_title="Business Toolkit", layout="wide")

# Sidebar
st.sidebar.title("Tools")
page = st.sidebar.radio("Choose a tool", ["Existing Customer", "New Customer"])

# Session State Initialization
initial_state = {
    "personas_df": None,
    "personas": [],
    "business_profile": None,
    "business_summary": "",
    "followup_questions": [],
    "competitor_videos": {},
    "video_comments": {},
    "comment_personas": [],
    "biz_defaults": {}
}
for key, val in initial_state.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------- Existing Customer ----------------
if page == "Existing Customer":
    st.title("Persona Builder (Existing Customer)")
    uploaded = st.file_uploader("Upload CSV (must include `customer_id`)", type="csv")

    col1, col2 = st.columns(2)
    with col1:
        generate_clicked = st.button("Generate Personas")
    with col2:
        if st.button("Reset Data"):
            for k in ["personas_df", "personas"]:
                st.session_state[k] = initial_state[k]
            st.rerun()

    if generate_clicked:
        if not uploaded:
            st.warning("Please upload a CSV file.")
            st.stop()

        df = pd.read_csv(uploaded)
        if "customer_id" not in df.columns:
            st.error("CSV missing required column: `customer_id`.")
            st.stop()

        df = df.fillna({col: "" if df[col].dtype == object else df[col].median() for col in df.columns})
        st.session_state.personas_df = df

        try:
            resp = requests.post(
                f"{API_BASE}/process_profiles",
                json={"profiles": df.to_dict(orient="records")},
                timeout=120
            )
            resp.raise_for_status()
            st.session_state.personas = resp.json().get("personas", [])
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")

    if st.session_state.personas_df is not None:
        st.subheader("Data Sample")
        st.dataframe(st.session_state.personas_df.head())

    if st.session_state.personas:
        st.success("Personas Generated")
        for persona in st.session_state.personas:
            st.json(persona)

# ---------------- New Customer ----------------
else:
    st.title("New Customer Onboarding")

    # Auto-fill inputs
    auto = st.checkbox("🌐 Auto-fill from website")
    if auto:
        website_url = st.text_input("Website URL", placeholder="https://example.com")
        max_pages = st.number_input("Max pages to scrape", min_value=1, max_value=100, value=25)
        max_workers = st.number_input("Number of workers (threads)", min_value=1, max_value=10, value=2)
        if st.button("Fetch data from site"):
            try:
                resp = requests.post(
                    f"{API_BASE}/extract_business_info",
                    json={"website_url": website_url, "max_pages": max_pages, "max_workers": max_workers},
                    timeout=120
                )
                resp.raise_for_status()
                st.session_state.biz_defaults = resp.json()
                st.success("Auto-fill data loaded. You can tweak the fields below.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching auto-fill data: {e}")

    # Prepare defaults
    raw = st.session_state.biz_defaults or {}
    defaults = {
        "name": raw.get("name", ""),
        "founded": raw.get("founded", ""),
        "locations": ", ".join(raw.get("locations", [])),
        "offerings": ", ".join(raw.get("offerings", [])),
        "price_range": raw.get("price_range", ""),
        "audience": ", ".join(raw.get("audience", [])),
        "usp": raw.get("usp", ""),
        "competitors": raw.get("competitors", []),
        "goals": "; ".join(raw.get("goals", []))
    }
    ui_channel_options = ["Email", "Social Media", "Events", "SEO", "Partnerships", "Paid Ads"]
    defaults["channels"] = [opt for opt in ui_channel_options if opt.lower() in [c.lower() for c in raw.get("channels", [])]]

    # Business profile form
    with st.form("biz_form"):
        name = st.text_input("Business Name", value=defaults["name"])
        raw_founded = defaults.get("founded", "")
        try:
            default_founded = int(raw_founded)
        except (ValueError, TypeError):
            default_founded = 2023
        founded = st.number_input("Year Founded", min_value=1900, max_value=2025, value=default_founded)
        locations = st.text_input("Location(s)", value=defaults["locations"], help="e.g. Mumbai; Pune")
        offerings = st.text_area("Products / Services (comma-separated)", value=defaults["offerings"])
        price_range = st.text_input("Price Range (e.g. ₹200–₹800)", value=defaults["price_range"])
        audience = st.text_area("Ideal Customers (demographics, region)", value=defaults["audience"])
        usp = st.text_area("Unique Selling Proposition", value=defaults["usp"])
        competitors = st.multiselect("Key Competitors", defaults["competitors"])
        channels = st.multiselect("Marketing Channels", ui_channel_options, default=defaults["channels"])
        goals = st.text_area("Top 3 Goals (semicolon-separated)", value=defaults["goals"])
        submitted = st.form_submit_button("Generate Business Profile")

    # Reset
    if st.button("Reset Data"):
        for k in initial_state.keys(): st.session_state[k] = initial_state[k]
        st.rerun()

    # Summarize
    if submitted:
        biz_payload = {"name": name, "founded": str(founded), "locations": locations,
                       "offerings": offerings, "price_range": price_range,
                       "audience": audience, "usp": usp,
                       "competitors": competitors, "channels": channels,
                       "goals": goals.split(";")}
        try:
            resp = requests.post(f"{API_BASE}/summarize_business", json={"business": biz_payload}, timeout=60)
            resp.raise_for_status(); st.session_state.business_profile = resp.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}"); st.stop()
        try:
            summ = requests.post(f"{API_BASE}/summarize_profile", json=st.session_state.business_profile, timeout=30)
            summ.raise_for_status(); st.session_state.business_summary = summ.json().get("summary", "")
        except requests.exceptions.RequestException as e:
            st.error(f"Summary API error: {e}")

    # Competitor insights
    if st.session_state.business_summary:
        st.subheader("Profile Summary"); st.write(st.session_state.business_summary)
        # Competitor questions
        if st.button("1. Generate Competitor Questions"):
            try:
                r = requests.post(f"{API_BASE}/generate_followup_queries",
                                   json={"summary": st.session_state.business_summary,
                                         "topic": "competitors",
                                         "competitors": st.session_state.business_profile.get("competitors", [])}, timeout=30)
                r.raise_for_status(); st.session_state.followup_questions = r.json().get("questions", [])
            except requests.exceptions.RequestException as e:
                st.error(f"Error generating questions: {e}")
        if st.session_state.followup_questions:
            st.markdown("**Competitor Follow-Up Questions:**")
            for i, q in enumerate(st.session_state.followup_questions,1): st.write(f"{i}. {q}")
            # Fetch videos
            if st.button("2. Fetch Competitor Videos"):
                vids_map = {}
                for comp in st.session_state.business_profile.get("competitors",[]):
                    try:
                        r_v = requests.post(f"{API_BASE}/youtube_search",
                                            json={"query":comp,"order":"viewCount","max_results":5},timeout=60)
                        r_v.raise_for_status(); vids_map[comp]=r_v.json().get("videos",[])
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error fetching videos for {comp}: {e}")
                st.session_state.competitor_videos = vids_map
            # Display videos
            if st.session_state.competitor_videos:
                st.subheader("Competitor Videos (Top 5 by Views)")
                for comp, vids in st.session_state.competitor_videos.items():
                    st.markdown(f"**{comp}**")
                    for v in vids: st.write(f"- [{v['title']}]({v['url']}) — {v['viewCount']} views")
            # Fetch comments
            if st.button("3. Fetch Video Comments"):
                all_ids=[vid['id'] for vids in st.session_state.competitor_videos.values() for vid in vids]
                try:
                    r_c= requests.post(f"{API_BASE}/youtube_comments_filtered",
                                       json={"video_ids":all_ids,"questions":st.session_state.followup_questions},timeout=120)
                    r_c.raise_for_status(); st.session_state.video_comments = r_c.json()
                except requests.exceptions.Timeout:
                    st.error("🕒 Fetching comments timed out. Try again.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error fetching filtered comments: {e}")
            # Display comments
            if st.session_state.video_comments:
                st.subheader("Top Semantically Relevant Comments")
                for vid, comms in st.session_state.video_comments.items():
                    st.write(f"**Video {vid}:**")
                    for c in comms: st.write(f"- {c}")
            # Generate personas
            if st.button("4. Generate Customer Personas"):
                try:
                    r_p = requests.post(
                        f"{API_BASE}/comment_personas",
                        json=st.session_state.video_comments,
                        timeout=60
                    )
                    r_p.raise_for_status(); st.session_state.comment_personas = r_p.json().get("personas", [])
                except requests.exceptions.RequestException as e:
                    st.error(f"Error generating customer personas: {e}")
            if st.session_state.comment_personas:
                st.subheader("Customer Personas from Competitor Insights")
                for persona in st.session_state.comment_personas: st.json(persona)
