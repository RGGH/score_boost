from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone

# --------------------------------------------------
# Freshness / time-decay configuration
# --------------------------------------------------

FRESHNESS_WEIGHT = 0.15
# Maximum amount freshness can add to the semantic score.
# Newest possible document ≈ +0.15
#
# Increase → freshness matters MORE
# Decrease → semantic relevance matters MORE


FRESHNESS_SCALE_DAYS = 30
# Time scale of the linear decay.
# the *reference distance* for shaping the decay
# It does not mean:
# "At day 30, turn the boost off."
#
# 30 means the freshness score decays over roughly 30 days.
#
# Increase → papers stay "fresh" for longer
# Decrease → papers become "old" more quickly


FRESHNESS_MIDPOINT = 0.3
# Controls the decay curve at the specified scale.
#
# At the configured scale, the decay value is calibrated
# around this midpoint.
#
# Higher → slower decay
# Lower → steeper decay


CANDIDATE_LIMIT = 100
# Number of semantic results considered before freshness
# re-ranks them.
#
# Larger → freshness has more candidates to promote
# Smaller → faster / less opportunity for re-ranking


RESULT_LIMIT = 10
# Number of final results returned.


# --------------------------------------------------
# Configuration
# --------------------------------------------------

COLLECTION_NAME = "arxiv_papers"

client = QdrantClient(
    host="localhost",
    port=6333,
)

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# Query
# --------------------------------------------------


def query(search_text):

    query_vector = model.encode(
        search_text,
        normalize_embeddings=True,
    ).tolist()

    now = datetime.now(timezone.utc)

    print()
    print("=" * 110)
    print(f"Query: {search_text}")
    print(f"Now:   {now.isoformat()}")
    print("=" * 110)

    # --------------------------------------------------
    # Semantic-only search
    # --------------------------------------------------

    semantic_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=100,
    )

    # --------------------------------------------------
    # Search with freshness formula
    # --------------------------------------------------

    formula_result = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=models.Prefetch(
            query=query_vector,
            limit=100,
        ),
        query=models.FormulaQuery(
            formula=models.SumExpression(
                sum=[
                    # --------------------------------------------------
                    # Original semantic similarity score
                    # --------------------------------------------------
                    "$score",
                    # --------------------------------------------------
                    # Freshness boost
                    #
                    # freshness_score × FRESHNESS_WEIGHT
                    # --------------------------------------------------
                    models.MultExpression(
                        mult=[
                            FRESHNESS_WEIGHT,
                            models.LinDecayExpression(
                                lin_decay=models.DecayParamsExpression(
                                    # Payload field containing the
                                    # publication Unix timestamp
                                    x="published_timestamp",
                                    # "Now" = reference point for age
                                    target=models.DatetimeExpression(
                                        datetime=now.isoformat(),
                                    ),
                                    # How quickly freshness decays
                                    scale=FRESHNESS_SCALE_DAYS * 86400,
                                    # Shape/calibration of the decay
                                    midpoint=FRESHNESS_MIDPOINT,
                                )
                            ),
                        ]
                    ),
                ]
            )
        ),
        limit=RESULT_LIMIT,
    )

    # --------------------------------------------------
    # Original semantic scores
    # --------------------------------------------------

    semantic_scores = {point.id: point.score for point in semantic_result.points}

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\nFINAL RESULTS")
    print("=" * 110)

    for rank, point in enumerate(formula_result.points, start=1):
        # payload = point.payload
        payload = point.payload or {}

        title = payload.get("title", "N/A")
        published = payload.get("published", "N/A")

        semantic_score = semantic_scores.get(point.id)
        final_score = point.score

        if semantic_score is not None:
            freshness_boost = final_score - semantic_score
        else:
            freshness_boost = 0.0

        # Green tick if freshness contributed anything
        freshness_indicator = "🟢" if freshness_boost > 0.000001 else ""

        # Calculate age for display
        timestamp = payload.get("published_timestamp")

        if timestamp is not None:
            age_days = (now.timestamp() - float(timestamp)) / 86400
        else:
            age_days = None

        print(f"\n#{rank} {freshness_indicator}")
        print("-" * 110)

        print(f"Title:             {title}")
        print(f"Published:         {published}")

        if age_days is not None:
            print(f"Age:               {age_days:.1f} days")
        else:
            print("Age:               N/A")

        print(f"Semantic score:    {semantic_score:.6f}")
        print(f"Freshness boost:   {freshness_boost:.6f}")
        print(f"FINAL score:       {final_score:.6f}")

        print(f"arXiv:             {payload.get('arxiv_id', 'N/A')}")
        print(f"URL:               {payload.get('url', 'N/A')}")


if __name__ == "__main__":
    search_text = input("Search: ")
    query(search_text)
