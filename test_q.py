#!/usr/bin/env python3
"""
test_qdrant.py

Quick connectivity check for your Qdrant Cloud cluster.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

def main():
    # 1) Load environment variables from .env
    load_dotenv()
    url     = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        print("❌ QDRANT_URL or QDRANT_API_KEY not set in environment")
        return

    print(f"→ Testing Qdrant Cloud at: {url}")

    # 2) Instantiate the client (REST only)
    try:
        client = QdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=False
        )
    except Exception as e:
        print(f"❌ Failed to create QdrantClient: {e!r}")
        return

    # 3) Try listing collections
    try:
        resp = client.get_collections().collections
        names = [c.name for c in resp]
        print("✅ Connected! Collections:", names)
    except Exception as e:
        print(f"❌ Connection failed when listing collections: {e!r}")

if __name__ == "__main__":
    main()
