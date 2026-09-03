from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone


# ==================================================
# Freshness / time-decay configuration
# ==================================================

FRESHNESS_WEIGHT = 0.15
# Maximum amount freshness can add to the semantic score.
#
# Increase → freshness matters MORE
# Decrease → semantic relevance matters MORE


FRESHNESS_SCALE_DAYS = 30
# Reference distance used by the linear decay.
#
# It does NOT mean:
# "At exactly 30 days, freshness becomes zero."
#
# Increase → papers stay fresh for longer
# Decrease → papers become old more quickly


FRESHNESS_MIDPOINT = 0.3
# Controls the decay curve at the configured scale.
#
# Higher → slower decay
# Lower → steeper decay


CANDIDATE_LIMIT = 100
# Number of semantic candidates considered by the
# Formula Query before freshness re-ranking.
#
# Larger → freshness has more candidates to promote
# Smaller → less work / less opportunity for re-ranking


RESULT_LIMIT = 10
# Number of final results returned.


# ==================================================
# Configuration
# ==================================================

COLLECTION_NAME = "arxiv_papers"

client = QdrantClient(
    host="localhost",
    port=6333,
)

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==================================================
# Payload index setup
# ==================================================

def ensure_payload_indexes():
    """
    Create indexes needed by Formula Query.

    published_timestamp is numeric, so use a FLOAT payload index.

    This should normally be run once after collection creation,
    rather than every time a query is performed.
    """

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="published_timestamp",
        field_schema=models.PayloadSchemaType.FLOAT,
    )


# ==================================================
# Query
# ==================================================

def query(search_text):

    # --------------------------------------------------
    # Generate query embedding
    # --------------------------------------------------

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
    #
    # Used for comparison/debugging.
    # --------------------------------------------------

    semantic_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=CANDIDATE_LIMIT,
        with_payload=True,
    )

    # --------------------------------------------------
    # Semantic search + freshness re-ranking
    # --------------------------------------------------

    formula_result = client.query_points(
        collection_name=COLLECTION_NAME,

        # ----------------------------------------------
        # Stage 1:
        # Retrieve semantic candidates.
        # ----------------------------------------------
        prefetch=models.Prefetch(
            query=query_vector,
            limit=CANDIDATE_LIMIT,
        ),

        # ----------------------------------------------
        # Stage 2:
        # Re-score those candidates using:
        #
        # final_score =
        #     semantic_score
        #     +
        #     freshness_boost
        # ----------------------------------------------
        query=models.FormulaQuery(

            formula=models.SumExpression(
                sum=[

                    # ----------------------------------
                    # Original semantic similarity score
                    # ----------------------------------
                    "$score",

                    # ----------------------------------
                    # Freshness boost
                    #
                    # FRESHNESS_WEIGHT
                    #       ×
                    # freshness_score
                    # ----------------------------------
                    models.MultExpression(
                        mult=[

                            FRESHNESS_WEIGHT,

                            models.LinDecayExpression(
                                lin_decay=models.DecayParamsExpression(

                                    # Payload field containing
                                    # Unix publication timestamp.
                                    x="published_timestamp",

                                    # "Now" against which age
                                    # is calculated.
                                    target=models.DatetimeExpression(
                                        datetime=now.isoformat(),
                                    ),

                                    # Decay scale in seconds.
                                    scale=FRESHNESS_SCALE_DAYS * 86400,

                                    # Shape/calibration of decay.
                                    midpoint=FRESHNESS_MIDPOINT,
                                )
                            ),
                        ]
                    ),
                ]
            ),

            # ----------------------------------------------
            # IMPORTANT:
            #
            # If a point has no published_timestamp,
            # Formula Query needs a defined fallback.
            #
            # 0.0 represents an old/zero freshness value,
            # so the document receives no useful freshness
            # boost rather than causing the formula to fail.
            # ----------------------------------------------
            defaults={
                "published_timestamp": 0.0,
            },
        ),

        limit=RESULT_LIMIT,
        with_payload=True,
    )

    # --------------------------------------------------
    # Original semantic scores
    # --------------------------------------------------

    semantic_scores = {
        point.id: point.score
        for point in semantic_result.points
    }

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\nFINAL RESULTS")
    print("=" * 110)

    for rank, point in enumerate(formula_result.points, start=1):

        payload = point.payload or {}

        title = payload.get("title", "N/A")
        published = payload.get("published", "N/A")

        semantic_score = semantic_scores.get(point.id)
        final_score = point.score

        # ----------------------------------------------
        # Calculate freshness contribution
        #
        # final = semantic + freshness
        #
        # Therefore:
        #
        # freshness = final - semantic
        # ----------------------------------------------

        if semantic_score is not None:
            freshness_boost = final_score - semantic_score
        else:
            freshness_boost = 0.0

        # Green tick if freshness contributed
        # a meaningful positive amount.
        freshness_indicator = (
            "🟢"
            if freshness_boost > 0.000001
            else ""
        )

        # ----------------------------------------------
        # Calculate document age for display.
        # ----------------------------------------------

        timestamp = payload.get("published_timestamp")

        if timestamp is not None:

            age_days = (
                now.timestamp() - float(timestamp)
            ) / 86400

        else:

            age_days = None

        # ----------------------------------------------
        # Display
        # ----------------------------------------------

        print(f"\n#{rank} {freshness_indicator}")
        print("-" * 110)

        print(f"Title:             {title}")
        print(f"Published:         {published}")

        if age_days is not None:
            print(f"Age:               {age_days:.1f} days")
        else:
            print("Age:               N/A")

        if semantic_score is not None:
            print(f"Semantic score:    {semantic_score:.6f}")
        else:
            print("Semantic score:    N/A")

        print(f"Freshness boost:   {freshness_boost:.6f}")
        print(f"FINAL score:       {final_score:.6f}")

        print(
            f"arXiv:             "
            f"{payload.get('arxiv_id', 'N/A')}"
        )

        print(
            f"URL:               "
            f"{payload.get('url', 'N/A')}"
        )


# ==================================================
# Main
# ==================================================

if __name__ == "__main__":

    # Create the required payload index.
    #
    # In production, do this once during collection
    # setup rather than every application startup.
    ensure_payload_indexes()

    search_text = input("Search: ")

    query(search_text)
