# upsert.py
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

# --------------------------------------------------
# 1. Load your CSV
# --------------------------------------------------
df = pd.read_csv("car_sales_data.csv")

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

COLLECTION_NAME = "car_sales_data"

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
# 5. Create text to embed
# --------------------------------------------------
# CSV columns: Manufacturer, Model, Engine size, Fuel type,
# Year of manufacture, Mileage, Price
texts = (
    "Manufacturer: " + df["Manufacturer"].astype(str)
    + "\nModel: " + df["Model"].astype(str)
    + "\nEngine size: " + df["Engine size"].astype(str) + "L"
    + "\nFuel type: " + df["Fuel type"].astype(str)
    + "\nYear of manufacture: " + df["Year of manufacture"].astype(str)
    + "\nMileage: " + df["Mileage"].astype(str) + " miles"
).tolist()

# --------------------------------------------------
# 6. Generate embeddings
# --------------------------------------------------
embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True,
)

# --------------------------------------------------
# 7. Create Qdrant points
# --------------------------------------------------
points = []
for i, row in enumerate(df.to_dict(orient="records")):
    payload = {
        # NOTE: keys below are lowercase/snake_case to match
        # what the search script's formula and print logic expect.
        "manufacturer": row["Manufacturer"],
        "model": row["Model"],
        "engine_size": float(row["Engine size"]),
        "fuel_type": row["Fuel type"],
        "year_of_manufacture": int(row["Year of manufacture"]),
        "mileage": int(row["Mileage"]),
        # Stored as a float since the price-proximity formula
        # (symmetric_rational_decay) does arithmetic on this field.
        "price": float(row["Price"]),
    }
    points.append(
        models.PointStruct(
            # CSV has no id column, so use the row index.
            id=i,
            vector=embeddings[i].tolist(),
            payload=payload,
        )
    )

# --------------------------------------------------
# 8. Upload to Qdrant in batches
# --------------------------------------------------
# A single upsert() call with all points can exceed Qdrant's
# default 32MB request payload limit for large datasets, so we
# chunk the upload instead.
UPSERT_BATCH_SIZE = 256

for start in range(0, len(points), UPSERT_BATCH_SIZE):
    batch = points[start : start + UPSERT_BATCH_SIZE]
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch,
    )
    print(f"Upserted {start + len(batch)}/{len(points)} cars...")

# --------------------------------------------------
# 9. Done
# --------------------------------------------------
print(f"Uploaded {len(points)} cars to Qdrant.")