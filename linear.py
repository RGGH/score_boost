
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone


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
                    "$score",

                    models.MultExpression(
                        mult=[
                            0.15,

                            models.LinDecayExpression(
                                lin_decay=models.DecayParamsExpression(
                                    x="published_timestamp",

                                    target=models.DatetimeExpression(
                                        datetime=now.isoformat(),
                                    ),

                                    scale=30 * 86400,

                                    midpoint=0.3,
                                )
                            ),
                        ]
                    ),
                ]
            )
        ),

        limit=10,
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
            age_days = (
                now.timestamp() - float(timestamp)
            ) / 86400
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
