import logging
import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

from .embeddings import upsert_embeddings
from .clustering import cluster_embeddings
from .persona_gen import generate_personas
from .business import summarize_business, summarize_profile
from .webfill import SmartWebScraper

# ─── New routers ───────────────────────────────────────────────────────────────
from .followup import router as followup_router
from .youtube_router import router as youtube_router
from .comment_personas import router as comment_personas_router

# ─── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Pydantic models ────────────────────────────────────────────────────────────
class Profile(BaseModel):
    customer_id: Any = Field(..., description="Unique customer identifier")
    class Config:
        extra = "allow"

class ProfilesRequest(BaseModel):
    profiles: List[Profile] = Field(..., description="List of customer profiles")

class PersonasResponse(BaseModel):
    personas: List[Dict[str, Any]] = Field(..., description="Generated persona objects")

class BizRequest(BaseModel):
    business: Dict[str, Any] = Field(..., description="Raw new-business input data")

class ExtractRequest(BaseModel):
    website_url: HttpUrl = Field(..., description="URL of the business website to scrape")
    max_pages:    int     = Field(25, description="Maximum number of pages to crawl")
    max_workers:  int     = Field(2,  description="Number of parallel worker threads")

# ─── FastAPI app setup ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Persona & Business Profile Service",
    description=(
        "Embed & cluster existing data into personas, "
        "summarize new-business inputs into profiles and human-readable summaries"
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}

# ─── Persona builder for existing data ─────────────────────────────────────────
@app.post(
    "/process_profiles",
    response_model=PersonasResponse,
    summary="Generate personas from existing customer data",
)
async def process_profiles(request: ProfilesRequest) -> PersonasResponse:
    profiles = [p.dict() for p in request.profiles]
    logger.info("Received %d profiles", len(profiles))
    try:
        triples  = upsert_embeddings(profiles)
        logger.info("Upserted %d embeddings", len(triples))

        clusters = cluster_embeddings(triples)
        logger.info("Formed %d clusters", len(clusters))

        personas = generate_personas(clusters)
        logger.info("Generated %d personas", len(personas))

        return PersonasResponse(personas=personas)
    except Exception:
        logger.exception("Error in /process_profiles")
        raise HTTPException(status_code=500, detail="Internal server error")

# ─── Summarize raw business into structured JSON ───────────────────────────────
@app.post(
    "/summarize_business",
    response_model=Dict[str, Any],
    summary="Summarize new-business owner inputs into a structured profile",
)
async def summarize_business_endpoint(request: BizRequest) -> Dict[str, Any]:
    logger.info("Received business input: %s", request.business)
    try:
        profile = summarize_business(request.business)
        logger.info("Structured profile: %s", profile)
        return profile
    except Exception as e:
        logger.exception("Error in /summarize_business")
        raise HTTPException(status_code=500, detail=str(e))

# ─── Website auto-fill endpoint ────────────────────────────────────────────────
@app.post(
    "/extract_business_info",
    response_model=Dict[str, Any],
    summary="Scrape a business website and auto-fill the profile fields",
)
async def extract_business_info(request: ExtractRequest) -> Dict[str, Any]:
    logger.info(
        "Extracting business info from URL=%s (pages=%d, workers=%d)",
        request.website_url, request.max_pages, request.max_workers
    )
    try:
        # Cast HttpUrl to str so that .rstrip() and other string methods work properly
        scraper = SmartWebScraper(
            base_url=str(request.website_url),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_workers=request.max_workers,
            max_pages=request.max_pages
        )
        results = scraper.run()
        if not results or "final_summary" not in results:
            logger.error("Scraper returned no summary or missing key")
            raise HTTPException(status_code=500, detail="Website analysis failed")

        # Map the free-form summary into your structured business profile
        structured = summarize_business({
            "summary": results["final_summary"]
        })
        logger.info("Auto-filled profile: %s", structured)
        return structured

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /extract_business_info")
        raise HTTPException(status_code=500, detail=str(e))

# ─── Human-friendly summary from structured profile ───────────────────────────
@app.post(
    "/summarize_profile",
    response_model=Dict[str, str],
    summary="Generate a human-readable summary from a structured business profile",
)
async def summarize_profile_endpoint(profile: Dict[str, Any]) -> Dict[str, str]:
    logger.info("Received profile for summarization: %s", profile)
    try:
        summary = summarize_profile(profile)
        logger.info("Generated summary: %s", summary)
        return {"summary": summary}
    except Exception as e:
        logger.exception("Error in /summarize_profile")
        raise HTTPException(status_code=500, detail=str(e))

# ─── Follow-up questions endpoint ──────────────────────────────────────────────
app.include_router(followup_router, prefix="", tags=["followup"])

# ─── YouTube comments endpoint ─────────────────────────────────────────────────
app.include_router(youtube_router, prefix="", tags=["youtube"])

# ─── Comment-based personas endpoint ──────────────────────────────────────────
app.include_router(comment_personas_router, prefix="", tags=["personas"])
