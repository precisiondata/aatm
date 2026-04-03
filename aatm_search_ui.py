import streamlit as st
import chromadb
import dotenv

from aatm.data_models import RetrievedExpressionMetadata, RetrieverResults
from aatm.retrievers import (
    CHROMADB_RETRIEVER_MODEL_REGISTRY,
    ChromaDBRetriever,
    load_chromadb_retriever,
)
from aatm.embedding_functions import GoogleEmbeddingFunction
from aatm.selectors import FirstResultSelector

dotenv.load_dotenv()

client = chromadb.PersistentClient()


@st.cache_resource
def cached_load_retriever(
    retriever_name: str = "embeddinggemma-300M",
) -> "ChromaDBRetriever":
    retriever = load_chromadb_retriever(retriever_name)
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
                result: RetrievedExpressionMetadata
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
