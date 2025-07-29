# backend/clustering.py

import logging
from typing import Any, Dict, List

import numpy as np
import hdbscan

# Configure module‐level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def cluster_embeddings(
    triples: List[Dict[str, Any]],
    min_cluster_fraction: float = 0.05,
    cluster_selection_epsilon: float = 0.0
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Dynamically cluster embeddings with HDBSCAN.

    Args:
        triples: List of dicts, each containing:
            - "id": unique identifier
            - "vector": List[float] embedding
            - "payload": original profile dict
        min_cluster_fraction: Minimum fraction of total points for a valid cluster
                              (e.g. 0.05 means clusters must have ≥5% of all points).
        cluster_selection_epsilon: Distance threshold to further split loose clusters.

    Returns:
        A dict mapping cluster_label -> list of payload dicts.
        Noise points (label == -1) are excluded.
    """
    n = len(triples)
    if n == 0:
        logger.warning("cluster_embeddings called with empty triples list.")
        return {}

    # 1) Build embeddings matrix
    try:
        vectors = np.vstack([t["vector"] for t in triples]).astype(float)
    except KeyError as e:
        logger.error("Missing 'vector' key in triples: %s", e)
        raise
    except Exception as e:
        logger.error("Error stacking vectors: %s", e)
        raise

    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D array of embeddings, got shape {vectors.shape}")

    # 2) Compute dynamic minimum cluster size
    min_cluster_size = max(2, int(n * min_cluster_fraction))
    logger.info(
        "Clustering %d points with min_cluster_size=%d (%.1f%% of data)",
        n, min_cluster_size, min_cluster_fraction * 100
    )

    # 3) Run HDBSCAN
    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric="euclidean"
        )
        labels = clusterer.fit_predict(vectors)
    except Exception as e:
        logger.error("HDBSCAN clustering failed: %s", e)
        raise

    # 4) Assemble clusters
    clusters: Dict[int, List[Dict[str, Any]]] = {}
    for triple, lbl in zip(triples, labels):
        if lbl == -1:  # skip noise
            continue
        clusters.setdefault(lbl, []).append(triple["payload"])

    # 5) Log summary: number of clusters and their sizes
    num_clusters = len(clusters)
    sizes = {label: len(members) for label, members in clusters.items()}
    logger.info("cluster_embeddings: found %d clusters", num_clusters)
    logger.info("cluster_embeddings: cluster sizes %s", sizes)

    return clusters
