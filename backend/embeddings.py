# backend/embeddings.py

import os
import logging
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException

# ─── Load env & configure logging ─────────────────────────────────────────────
load_dotenv()
HF_TOKEN        = os.getenv("HF_TOKEN")
QDRANT_URL      = os.getenv("QDRANT_URL")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set in the environment")
if not QDRANT_URL:
    raise ValueError("QDRANT_URL is not set in the environment")
if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY is not set in the environment")

# Export Hugging Face token for the embedder
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ─── Initialize the HF embedding client ──────────────────────────────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
logger.info("Loaded HuggingFaceEmbeddings model '%s'", EMBED_MODEL)


def get_qdrant_client() -> QdrantClient:
    """
    Instantiate a QdrantClient for Cloud (REST).
    """
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=False
    )


def ensure_profiles_collection(client: QdrantClient) -> None:
    """
    Create 'profiles' collection if missing.
    """
    try:
        existing = [c.name for c in client.get_collections().collections]
    except (UnexpectedResponse, ResponseHandlingException) as e:
        logger.error("Could not list Qdrant collections: %s", e)
        raise RuntimeError("Cannot contact Qdrant at startup") from e

    if "profiles" not in existing:
        client.create_collection(
            collection_name="profiles",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection 'profiles' in the cloud.")


def upsert_embeddings(profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    1) Serialize each profile (excluding 'customer_id') to text.
    2) Get embeddings via HF Inference.
    3) Lazily init Qdrant & ensure 'profiles' collection exists.
    4) Upsert to Qdrant.
    5) Return triples for clustering.
    """
    # Build texts & IDs
    texts = [
        " | ".join(f"{k}: {v}" for k, v in p.items() if k != "customer_id")
        for p in profiles
    ]
    ids = [p["customer_id"] for p in profiles]

    # Compute embeddings
    vectors = embedder.embed_documents(texts)

    # Pack points & triples
    points: List[Dict[str, Any]] = []
    triples: List[Dict[str, Any]] = []
    for pid, vec, payload in zip(ids, vectors, profiles):
        point = {"id": pid, "vector": vec, "payload": payload}
        points.append(point)
        triples.append(point.copy())

    if not points:
        logger.warning("No profiles to upsert.")
        return []

    # Connect & ensure collection
    client = get_qdrant_client()
    ensure_profiles_collection(client)

    # Upsert
    try:
        client.upsert(collection_name="profiles", points=points)
        logger.info("Upserted %d embeddings to Qdrant Cloud 'profiles'.", len(points))
    except Exception as e:
        logger.error("Failed to upsert to Qdrant: %s", e)
        raise

    return triples
