"""
load_to_neo4j.py
----------------
One-shot ETL script: loads a curated subset of Facebook friend-candidate
pairs (with Random Forest–predicted probabilities) into a Neo4j Aura instance.

Run once locally with the venv active:
    python load_to_neo4j.py

Credentials are read from .streamlit/secrets.toml (same file the Streamlit
app uses) via neo4j_db._get_secret(), with env-var fall-back.

Safe to re-run: MERGE semantics make every operation idempotent.
"""

from __future__ import annotations

# ── tuneable constant ─────────────────────────────────────────────────────────
EXTRA_PAIRS: int = 5000          # balanced extra pairs (half pos, half neg)
RANDOM_SEED: int = 42
BATCH_SIZE:  int = 500           # rows per UNWIND batch

# ── edge properties to carry into Neo4j ──────────────────────────────────────
EDGE_FEATURE_COLS = [
    "jaccard_followers",
    "jaccard_followees",
    "adar_index",
    "inter_followers",
    "follows_back",
    "shortest_path",
]

# ── columns dropped before model.predict_proba (same as recommend_friends()) ─
DROP_COLS = ["source_node", "destination_node", "indicator_link"]

# ─────────────────────────────────────────────────────────────────────────────
import gzip
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from neo4j import GraphDatabase

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("load_to_neo4j")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Read credentials (reuse neo4j_db helpers, but without Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def _read_secrets_toml() -> dict[str, str]:
    """Parse .streamlit/secrets.toml manually (no Streamlit runtime needed)."""
    secrets: dict[str, str] = {}
    toml_path = Path(".streamlit") / "secrets.toml"
    if not toml_path.exists():
        return secrets
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            # Fallback: use the toml package installed as a streamlit dep
            import toml  # type: ignore[import-untyped]
            with toml_path.open() as fh:
                return toml.load(fh)
    with toml_path.open("rb") as fh:
        return tomllib.load(fh)


def _get_cred(key: str, default: str = "") -> str:
    import os
    secrets = _read_secrets_toml()
    return str(secrets.get(key, os.environ.get(key, default)))


def get_driver():
    uri      = _get_cred("NEO4J_URI")
    username = _get_cred("NEO4J_USERNAME", "neo4j")
    password = _get_cred("NEO4J_PASSWORD")
    if not uri or not password:
        log.error(
            "NEO4J_URI / NEO4J_PASSWORD not found in secrets.toml or env vars."
        )
        sys.exit(1)
    log.info("Connecting to %s as '%s' …", uri, username)
    return GraphDatabase.driver(uri, auth=(username, password))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load data files (exactly as frontend.py does)
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Loading top-30 nodes list …")
    top30 = pd.read_csv("top_30_warm_start_analysis_2plus_friends.csv")

    log.info("Loading featured_test_data.csv.gz (may take a moment) …")
    t0 = time.time()
    df = pd.read_csv("featured_test_data.csv.gz", compression="gzip")
    log.info("  Loaded %d rows in %.1fs", len(df), time.time() - t0)
    return top30, df


def load_model():
    log.info("Loading Random Forest model …")
    with gzip.open("best_random_forest_model_compressed.pkl.gz", "rb") as fh:
        model = pickle.load(fh)
    log.info("  Model loaded (%s)", type(model).__name__)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build the demo subset
# ─────────────────────────────────────────────────────────────────────────────

def build_subset(top30: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    top_node_ids = set(top30["node_id"].astype(int).tolist())

    # All rows that mention any top-30 node as source
    mask_top30 = df["source_node"].isin(top_node_ids)
    df_top30 = df[mask_top30].copy()
    log.info("  Pairs involving top-30 nodes (source): %d", len(df_top30))

    # Extra balanced pairs from the remainder
    df_rest = df[~mask_top30]
    n_each = EXTRA_PAIRS // 2

    rng = np.random.default_rng(RANDOM_SEED)

    pos_idx = df_rest.index[df_rest["indicator_link"] == 1].tolist()
    neg_idx = df_rest.index[df_rest["indicator_link"] == 0].tolist()

    n_pos = min(n_each, len(pos_idx))
    n_neg = min(n_each, len(neg_idx))

    sampled_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    sampled_neg = rng.choice(neg_idx, size=n_neg, replace=False)

    df_extra = df_rest.loc[np.concatenate([sampled_pos, sampled_neg])].copy()
    log.info("  Extra balanced pairs: %d pos + %d neg = %d",
             n_pos, n_neg, len(df_extra))

    subset = pd.concat([df_top30, df_extra], ignore_index=True)

    # Tag top-node sources
    subset["is_top_node"] = subset["source_node"].isin(top_node_ids)

    log.info("  Final subset: %d pairs", len(subset))
    return subset


# ─────────────────────────────────────────────────────────────────────────────
# 4. Score with the model
# ─────────────────────────────────────────────────────────────────────────────

def score_subset(subset: pd.DataFrame, model) -> pd.DataFrame:
    log.info("Running predict_proba on %d rows …", len(subset))
    X = subset.drop(columns=DROP_COLS + ["is_top_node"])
    probs = model.predict_proba(X)[:, 1]
    subset = subset.copy()
    subset["probability"] = probs
    subset["confidence"] = pd.cut(
        probs,
        bins=[-np.inf, 0.5, 0.7, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    log.info("  Scoring done.")
    return subset


# ─────────────────────────────────────────────────────────────────────────────
# 5. Neo4j – schema setup
# ─────────────────────────────────────────────────────────────────────────────

CONSTRAINT_CYPHER = """
CREATE CONSTRAINT user_id_unique IF NOT EXISTS
FOR (u:User) REQUIRE u.id IS UNIQUE
"""

MERGE_CYPHER = """
UNWIND $rows AS row
MERGE (src:User {id: row.source_node})
  ON CREATE SET src.is_top_node = row.is_top_node
  ON MATCH  SET src.is_top_node = src.is_top_node OR row.is_top_node
MERGE (dst:User {id: row.destination_node})
MERGE (src)-[r:CANDIDATE]->(dst)
  ON CREATE SET
    r.probability        = row.probability,
    r.confidence         = row.confidence,
    r.actual             = row.actual,
    r.jaccard_followers  = row.jaccard_followers,
    r.jaccard_followees  = row.jaccard_followees,
    r.adar_index         = row.adar_index,
    r.inter_followers    = row.inter_followers,
    r.follows_back       = row.follows_back,
    r.shortest_path      = row.shortest_path
  ON MATCH SET
    r.probability        = row.probability,
    r.confidence         = row.confidence,
    r.actual             = row.actual
"""

COUNT_CYPHER_NODES = "MATCH (u:User) RETURN count(u) AS n"
COUNT_CYPHER_RELS  = "MATCH ()-[r:CANDIDATE]->() RETURN count(r) AS n"


def setup_schema(session) -> None:
    log.info("Creating uniqueness constraint on (:User {id}) …")
    session.run(CONSTRAINT_CYPHER)
    log.info("  Constraint ready.")


def load_batches(session, subset: pd.DataFrame) -> None:
    # Build row dicts with only the fields the Cypher uses
    needed_cols = [
        "source_node", "destination_node", "is_top_node",
        "probability", "confidence", "indicator_link",
    ] + EDGE_FEATURE_COLS

    rows_df = subset[needed_cols].copy()
    rows_df.rename(columns={"indicator_link": "actual"}, inplace=True)

    # Convert numpy types to plain Python for the driver
    records = rows_df.to_dict(orient="records")
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, (np.integer,)):
                rec[k] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                rec[k] = bool(v)

    total = len(records)
    loaded = 0
    log.info("Writing %d rows to Neo4j in batches of %d …", total, BATCH_SIZE)

    for start in range(0, total, BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        session.run(MERGE_CYPHER, parameters={"rows": batch})
        loaded += len(batch)
        pct = loaded / total * 100
        log.info("  %d / %d  (%.0f%%)", loaded, total, pct)

    log.info("  All batches written.")


def print_counts(session) -> None:
    n_nodes = session.run(COUNT_CYPHER_NODES).single()["n"]
    n_rels  = session.run(COUNT_CYPHER_RELS).single()["n"]
    print()
    print("=" * 50)
    print(f"  (:User) nodes        : {n_nodes:>10,}")
    print(f"  [:CANDIDATE] edges   : {n_rels:>10,}")
    print("=" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== load_to_neo4j.py  (EXTRA_PAIRS=%d) ===", EXTRA_PAIRS)

    # Data
    top30, df = load_data()
    model     = load_model()
    subset    = build_subset(top30, df)
    subset    = score_subset(subset, model)

    # Neo4j
    driver = get_driver()
    db     = _get_cred("NEO4J_DATABASE", "neo4j")

    try:
        driver.verify_connectivity()
        log.info("Connection verified.")
    except Exception as exc:
        log.error("Cannot reach Neo4j: %s", exc)
        driver.close()
        sys.exit(1)

    with driver.session(database=db) as session:
        setup_schema(session)
        load_batches(session, subset)
        print_counts(session)

    driver.close()
    log.info("Done. Driver closed.")


if __name__ == "__main__":
    main()
