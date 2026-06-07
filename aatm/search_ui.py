"""
Provide a Streamlit-based search interface for the Any-to-Any Terminology Mapper.

This module defines an interactive web application for querying a terminology
retrieval system backed by ChromaDB. It exposes a simple search UI with
multiple tabs, supports cached retriever loading, and displays ranked candidate
concepts with key metadata and confidence scores.

The application is intended for local or internal use as a lightweight search
front end for terminology exploration and validation.
"""

import streamlit as st
import chromadb
import dotenv

from aatm.registries.retrievers import CHROMADB_RETRIEVER_MODEL_REGISTRY, load_retriever
from aatm.retrievers import (
    ChromaDBRetriever,
)

dotenv.load_dotenv()

client = chromadb.PersistentClient()


@st.cache_resource
def cached_load_retriever(
    retriever_name: str = "embeddinggemma-300M",
) -> "ChromaDBRetriever":
    """Load and cache a retriever instance for the Streamlit application.

    This function wraps retriever loading with Streamlit's resource cache so
    that the same retriever is reused across reruns, avoiding repeated model or
    database initialization overhead.

    Args:
        retriever_name: Name of the retriever configuration to load from the
            retriever registry.

    Returns:
        A loaded ``ChromaDBRetriever`` instance corresponding to the requested
            retriever name.

    Raises:
        KeyError: If the provided retriever name is not found in the registry.
        Exception: Propagates errors raised during retriever initialization.
    """
    retriever = load_retriever(retriever_name)
    return retriever


retriever = cached_load_retriever()

st.header("Any-to-Any Terminology Mapper")
st.subheader("Search engine")

search_an_expression_tab, search_a_concept_id_tab, search_a_vocabulary_code_tab = (
    st.tabs(["Search an expression", "Search a concept id", "Search a vocabulary code"])
)

with search_an_expression_tab:
    with st.form("search_an_expression_tab_form", border=False):
        with st.sidebar:
            retriever_name = st.selectbox(
                "Retriever", sorted(CHROMADB_RETRIEVER_MODEL_REGISTRY.keys())
            )
            retriever = cached_load_retriever(retriever_name)
        col1, col2 = st.columns([8, 2], vertical_alignment="bottom")
        with col1:
            query = st.text_input(
                "Terminology search engine",
                placeholder="Enter search term here...",
                label_visibility="hidden",
                key="search_an_expression_tab_input",
            )
        with col2:
            submitted = st.form_submit_button("Search", use_container_width=True)
        if submitted:
            results = query | retriever
            for idx, result in enumerate(results.results[0]):
                expander = st.expander(f"{idx + 1} - {result.expression.capitalize()}")
                with expander:
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.write(f"**Concept name:** {result.std_concept_name}")
                        st.write(f"**Standard concept id:** {result.std_concept_id}")
                        st.write(
                            f"**Standard vocabulary:** {result.std_vocabulary_id.value}"
                        )
                        st.write(
                            f"**Standard vocabulary code:** {result.std_vocabulary_code}"
                        )
                    with col2:
                        st.metric(
                            "Confidence score", value=f"{1 - result.distance:.2f}"
                        )

with search_a_concept_id_tab:
    with st.form("search_a_concept_id_tab_form", border=False):
        col1, col2 = st.columns([8, 2], vertical_alignment="bottom")
        with col1:
            query = st.text_input(
                "Terminology search engine",
                placeholder="Enter search term here...",
                label_visibility="hidden",
                key="search_a_concept_id_tab_input",
            )
        with col2:
            submitted = st.form_submit_button("Search", use_container_width=True)
        if submitted:
            pass

with search_a_vocabulary_code_tab:
    with st.form("search_a_vocabulary_code_tab_form", border=False):
        col1, col2 = st.columns([8, 2], vertical_alignment="bottom")
        with col1:
            query = st.text_input(
                "Terminology search engine",
                placeholder="Enter search term here...",
                label_visibility="hidden",
                key="search_a_vocabulary_code_tab_input",
            )
        with col2:
            submitted = st.form_submit_button("Search", use_container_width=True)
        if submitted:
            pass
