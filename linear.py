import streamlit as st
import pandas as pd
from datetime import datetime, timezone

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Freshness-Aware Semantic Search", layout="wide")

# --------------------------------------------------
# Cached resources (loaded once, reused across runs)
# --------------------------------------------------


@st.cache_resource
def get_client(host, port):
    return QdrantClient(host=host, port=port)


@st.cache_resource
def get_model(name):
    return SentenceTransformer(name)


# --------------------------------------------------
# Core query logic (same behavior as the CLI script)
# --------------------------------------------------


def run_query(
    client,
    model,
    collection_name,
    search_text,
    freshness_weight,
    freshness_scale_days,
    freshness_midpoint,
    candidate_limit,
    result_limit,
):
    query_vector = model.encode(search_text, normalize_embeddings=True).tolist()
    now = datetime.now(timezone.utc)

    semantic_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=candidate_limit,
    )

    formula_result = client.query_points(
        collection_name=collection_name,
        prefetch=models.Prefetch(query=query_vector, limit=candidate_limit),
        query=models.FormulaQuery(
            formula=models.SumExpression(
                sum=[
                    "$score",
                    models.MultExpression(
                        mult=[
                            freshness_weight,
                            models.LinDecayExpression(
                                lin_decay=models.DecayParamsExpression(
                                    x="published_timestamp",
                                    target=models.DatetimeExpression(datetime=now.isoformat()),
                                    scale=freshness_scale_days * 86400,
                                    midpoint=freshness_midpoint,
                                )
                            ),
                        ]
                    ),
                ]
            )
        ),
        limit=result_limit,
    )

    semantic_scores = {point.id: point.score for point in semantic_result.points}

    rows = []
    for rank, point in enumerate(formula_result.points, start=1):
        payload = point.payload or {}

        title = payload.get("title", "N/A")
        published = payload.get("published", "N/A")

        semantic_score = semantic_scores.get(point.id)
        final_score = point.score
        freshness_boost = (final_score - semantic_score) if semantic_score is not None else 0.0

        timestamp = payload.get("published_timestamp")
        age_days = (now.timestamp() - float(timestamp)) / 86400 if timestamp is not None else None

        rows.append(
            {
                "Rank": rank,
                "🟢": "🟢" if freshness_boost > 1e-6 else "",
                "Title": title,
                "Published": published,
                "Age (days)": round(age_days, 1) if age_days is not None else None,
                "Semantic score": round(semantic_score, 6) if semantic_score is not None else None,
                "Freshness boost": round(freshness_boost, 6),
                "Final score": round(final_score, 6),
                "arXiv": payload.get("arxiv_id", "N/A"),
                "URL": payload.get("url", "N/A"),
            }
        )

    return pd.DataFrame(rows), now


# --------------------------------------------------
# Sidebar: connection + constants as sliders
# --------------------------------------------------

st.sidebar.header("Connection")
host = st.sidebar.text_input("Qdrant host", value="localhost")
port = st.sidebar.number_input("Qdrant port", value=6333, step=1)
collection_name = st.sidebar.text_input("Collection name", value="arxiv_papers")
model_name = st.sidebar.text_input("Embedding model", value="all-MiniLM-L6-v2")

st.sidebar.header("Freshness / decay constants")

freshness_weight = st.sidebar.slider(
    "FRESHNESS_WEIGHT",
    min_value=0.0,
    max_value=1.0,
    value=0.15,
    step=0.01,
    help="Max amount freshness can add to the semantic score. Higher = freshness matters more.",
)

freshness_scale_days = st.sidebar.slider(
    "FRESHNESS_SCALE_DAYS",
    min_value=1,
    max_value=365,
    value=30,
    step=1,
    help="Reference time scale of the decay. Higher = papers stay 'fresh' longer.",
)

freshness_midpoint = st.sidebar.slider(
    "FRESHNESS_MIDPOINT",
    min_value=0.01,
    max_value=0.99,
    value=0.30,
    step=0.01,
    help="Decay value at the configured scale. Higher = slower decay, lower = steeper decay.",
)

candidate_limit = st.sidebar.slider(
    "CANDIDATE_LIMIT",
    min_value=10,
    max_value=500,
    value=100,
    step=10,
    help="Number of semantic results considered before freshness re-ranks them.",
)

result_limit = st.sidebar.slider(
    "RESULT_LIMIT",
    min_value=1,
    max_value=50,
    value=10,
    step=1,
    help="Number of final results returned.",
)

# --------------------------------------------------
# Main panel
# --------------------------------------------------

st.title("🔎 Freshness-Aware Semantic Search")
st.caption("Semantic search over arXiv papers with a tunable recency boost, backed by Qdrant.")

search_text = st.text_input("Search query", placeholder="e.g. reinforcement learning from human feedback")
search_clicked = st.button("Search", type="primary")

if search_clicked:
    if not search_text.strip():
        st.warning("Enter a search query first.")
    else:
        try:
            with st.spinner("Loading model and connecting to Qdrant..."):
                client = get_client(host, port)
                model = get_model(model_name)

            with st.spinner("Running search..."):
                df, now = run_query(
                    client,
                    model,
                    collection_name,
                    search_text,
                    freshness_weight,
                    freshness_scale_days,
                    freshness_midpoint,
                    candidate_limit,
                    result_limit,
                )

            st.caption(f"Query time (UTC): {now.isoformat()}")

            if df.empty:
                st.info("No results found.")
            else:
                st.dataframe(
                    df,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        "URL": st.column_config.LinkColumn("URL"),
                    },
                )
        except Exception as e:
            st.error(f"Search failed: {e}")
else:
    st.info("Set your query and constants, then click **Search**.")

