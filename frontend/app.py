# frontend/app.py

import streamlit as st
import pandas as pd
import requests

API_BASE = "http://localhost:8000"
st.set_page_config(page_title="Business Toolkit", layout="wide")

# Sidebar
st.sidebar.title("Tools")
page = st.sidebar.radio("Choose a tool", ["Existing Customer", "New Customer"])

# Session State Initialization
defaults = {
    "personas_df": None,
    "personas": [],
    "business_profile": None,
    "business_summary": "",
    "followup_questions": [],
    "youtube_comments": {},
    "comment_personas": []
}
for key, val in defaults.items():
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
            st.session_state.personas_df = None
            st.session_state.personas = []
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
    with st.form("biz_form"):
        name = st.text_input("Business Name")
        founded = st.number_input("Year Founded", min_value=1900, max_value=2025, value=2023)
        locations = st.text_input("Location(s)", help="e.g. Mumbai; Pune")
        offerings = st.text_area("Products / Services (comma-separated)")
        price_range = st.text_input("Price Range (e.g. ₹200–₹800)")
        audience = st.text_area("Ideal Customers (demographics, region)")
        usp = st.text_area("Unique Selling Proposition")
        competitors = st.text_area("Key Competitors")
        channels = st.multiselect(
            "Marketing Channels",
            ["Email", "Social Media", "Events", "SEO", "Partnerships", "Paid Ads"]
        )
        goals = st.text_area("Top 3 Goals for Next Year")
        submitted = st.form_submit_button("Generate Business Profile")

    if st.button("Reset Data"):
        st.session_state.business_profile = None
        st.session_state.business_summary = ""
        st.session_state.followup_questions = []
        st.session_state.youtube_comments = {}
        st.session_state.comment_personas = []
        st.rerun()

    # Step 1: Generate business profile & summary
    if submitted:
        biz = {
            "name": name,
            "founded": str(founded),
            "locations": locations,
            "offerings": offerings,
            "price_range": price_range,
            "audience": audience,
            "usp": usp,
            "competitors": competitors,
            "channels": channels,
            "goals": goals
        }
        try:
            resp = requests.post(
                f"{API_BASE}/summarize_business",
                json={"business": biz},
                timeout=60
            )
            resp.raise_for_status()
            st.session_state.business_profile = resp.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            st.stop()

        try:
            summ = requests.post(
                f"{API_BASE}/summarize_profile",
                json=st.session_state.business_profile,
                timeout=30
            )
            summ.raise_for_status()
            st.session_state.business_summary = summ.json().get("summary", "")
        except requests.exceptions.RequestException as e:
            st.error(f"Summary API error: {e}")

    # Display structured profile and summary
    if st.session_state.business_profile:
        st.success("Structured Business Profile")
        st.json(st.session_state.business_profile)

    if st.session_state.business_summary:
        st.subheader("Profile Summary")
        st.write(st.session_state.business_summary)

        # Step 2: Generate Follow-Up Questions
        if st.button("1. Generate Follow-Up Questions"):
            try:
                r1 = requests.post(
                    f"{API_BASE}/generate_followup_queries",
                    json={"summary": st.session_state.business_summary}
                )
                r1.raise_for_status()
                st.session_state.followup_questions = r1.json().get("questions", [])
            except requests.exceptions.RequestException as e:
                st.error(f"Error generating questions: {e}")

        if st.session_state.followup_questions:
            st.markdown("**Follow-Up Questions:**")
            for i, q in enumerate(st.session_state.followup_questions, 1):
                st.write(f"{i}. {q}")

            # Step 3: Fetch YouTube Comments
            if st.button("2. Fetch YouTube Comments"):
                try:
                    r2 = requests.post(
                        f"{API_BASE}/youtube_comments",
                        json={"questions": st.session_state.followup_questions},
                        timeout=60
                    )
                    r2.raise_for_status()
                    st.session_state.youtube_comments = r2.json()
                except requests.exceptions.RequestException as e:
                    st.error(f"Error fetching YouTube comments: {e}")

        # Display YouTube comments
        if st.session_state.youtube_comments:
            st.subheader("YouTube Comments")
            for question, comments in st.session_state.youtube_comments.items():
                st.markdown(f"**{question}**")
                for c in comments:
                    st.write(f"- {c}")

            # Step 4: Generate Comment Personas
            if st.button("3. Generate Comment-Personas"):
                try:
                    r3 = requests.post(
                        f"{API_BASE}/comment_personas",
                        json=st.session_state.youtube_comments,
                        timeout=60
                    )
                    r3.raise_for_status()
                    st.session_state.comment_personas = r3.json().get("personas", [])
                except requests.exceptions.RequestException as e:
                    st.error(f"Error generating comment personas: {e}")

        # Display generated comment personas
        if st.session_state.comment_personas:
            st.subheader("Personas from YouTube Feedback")
            for persona in st.session_state.comment_personas:
                st.json(persona)
