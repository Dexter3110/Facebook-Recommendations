"""
neo4j_db.py
-----------
Provides a cached Neo4j driver, a parameterised query runner, and a
connectivity probe for the Facebook Friend Recommendation Streamlit app.

Credentials are read from (in priority order):
  1. st.secrets  – set via .streamlit/secrets.toml (local) or
                   Streamlit Cloud -> App Settings -> Secrets.
  2. Environment variables – so plain Python scripts (notebooks, CLI
                             tools) can reuse this module without Streamlit.

All public functions silently swallow every exception; the app continues
to work in full when Neo4j is unreachable or credentials are absent.
"""

from __future__ import annotations

import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------

def _get_secret(key: str, default: str = "") -> str:
    """Return a secret value from st.secrets or an environment variable."""
    # Try Streamlit secrets first (works when running under `streamlit run`)
    try:
        import streamlit as st  # noqa: PLC0415  (late import is intentional)
        value = st.secrets.get(key, "")
        if value:
            return str(value)
    except Exception:
        pass  # Streamlit not initialised (e.g. plain script / notebook)

    # Fall back to environment variables
    return os.environ.get(key, default)


def _get_connection_params() -> dict[str, str]:
    return {
        "uri":      _get_secret("NEO4J_URI"),
        "username": _get_secret("NEO4J_USERNAME", "neo4j"),
        "password": _get_secret("NEO4J_PASSWORD"),
        "database": _get_secret("NEO4J_DATABASE", "neo4j"),
    }


# ---------------------------------------------------------------------------
# Cached driver (Streamlit-aware)
# ---------------------------------------------------------------------------

def _make_driver():
    """
    Create and return a neo4j.GraphDatabase.driver instance.

    Returns None if the neo4j package is missing or credentials are absent.
    """
    try:
        from neo4j import GraphDatabase  # noqa: PLC0415
    except ImportError:
        logger.warning("neo4j package not installed - Neo4j features disabled.")
        return None

    params = _get_connection_params()
    if not params["uri"] or not params["password"]:
        logger.warning(
            "NEO4J_URI / NEO4J_PASSWORD not set - Neo4j features disabled."
        )
        return None

    driver = GraphDatabase.driver(
        params["uri"],
        auth=(params["username"], params["password"]),
    )
    return driver


def get_driver():
    """
    Return a cached Neo4j driver when running inside Streamlit,
    or create a fresh one when running as a plain script.

    Returns None if the driver cannot be created.
    """
    try:
        import streamlit as st  # noqa: PLC0415

        # Wrap _make_driver with st.cache_resource lazily so this module can
        # still be imported outside Streamlit without error.
        if not hasattr(get_driver, "_cached"):
            get_driver._cached = st.cache_resource(show_spinner=False)(
                _make_driver
            )
        return get_driver._cached()
    except Exception:
        pass

    # Outside Streamlit - just build the driver directly.
    try:
        return _make_driver()
    except Exception as exc:
        logger.error("Failed to create Neo4j driver: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_query(
    cypher: str,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Execute a read query and return results as a list of plain dicts.

    Parameters
    ----------
    cypher : str
        A parameterised Cypher query string (never interpolate user data).
    params : dict, optional
        Named parameters bound to the Cypher query.

    Returns
    -------
    list[dict]
        Each element is one result record converted to a plain Python dict.
        Returns an empty list on any error (driver unavailable, query error,
        network failure, etc.).
    """
    try:
        driver = get_driver()
        if driver is None:
            return []

        db_name = _get_connection_params()["database"] or "neo4j"
        with driver.session(database=db_name) as session:
            result = session.run(cypher, parameters=params or {})
            return [dict(record) for record in result]
    except Exception as exc:
        logger.error("Neo4j query error: %s", exc)
        return []


def is_connected() -> bool:
    """
    Return True if a live Neo4j connection can be verified, False otherwise.

    Never raises; always returns a bool so it is safe to call from UI code.
    """
    try:
        driver = get_driver()
        if driver is None:
            return False
        driver.verify_connectivity()
        return True
    except Exception:
        return False
