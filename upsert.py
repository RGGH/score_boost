# upsert.py
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from datetime import timezone, datetime


# --------------------------------------------------
# 1. Load your CSV
# --------------------------------------------------

df = pd.read_csv("arxiv_papers.csv")

# Replace NaN values with empty strings
df = df.fillna("")


# --------------------------------------------------
# 2. Load embedding model
# --------------------------------------------------

# Produces 384-dimensional vectors
model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 3. Create Qdrant client
# --------------------------------------------------

client = QdrantClient(
    host="localhost",
    port=6333,
)

COLLECTION_NAME = "arxiv_papers"


# --------------------------------------------------
# 4. Create collection
# --------------------------------------------------

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
        ),
    )


# --------------------------------------------------
# 5. Convert 'published' date to Unix timestamp
# --------------------------------------------------


def to_timestamp(value) -> float | None:
    """
    Convert an ISO date/time string to a Unix timestamp.

    Example:
        2024-03-15T12:30:00Z
        -> 1710505800.0
    """
    value = str(value).strip()

    if not value:
        return None

    value = value.replace("Z", "+00:00")

    dt = datetime.fromisoformat(value)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.timestamp()


# --------------------------------------------------
# 6. Create text to embed
# --------------------------------------------------

texts = (
    "Title: "
    + df["title"].astype(str)
    + "\nDescription: "
    + df["description"].astype(str)
    + "\nAbstract: "
    + df["abstract"].astype(str)
).tolist()


# --------------------------------------------------
# 7. Generate embeddings
# --------------------------------------------------

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True,
)


# --------------------------------------------------
# 8. Create Qdrant points
# --------------------------------------------------

points = []

for i, row in enumerate(df.to_dict(orient="records")):
    published_timestamp = to_timestamp(row["published"])

    payload = {
        "arxiv_id": row["arxiv_id"],
        "url": row["url"],
        "title": row["title"],
        "description": row["description"],
        "abstract": row["abstract"],
        "published": row["published"],
        "published_timestamp": published_timestamp,
        "bucket_month": row["bucket_month"],
    }

    # Don't include a missing timestamp
    # because the FormulaQuery expects a number.
    if published_timestamp is None:
        payload.pop("published_timestamp")

    points.append(
        models.PointStruct(
            id=int(row["id"]),
            vector=embeddings[i].tolist(),
            payload=payload,
        )
    )


# --------------------------------------------------
# 9. Upload to Qdrant
# --------------------------------------------------

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
)


# --------------------------------------------------
# 10. Done
# --------------------------------------------------

print(f"Uploaded {len(points)} papers to Qdrant.")
