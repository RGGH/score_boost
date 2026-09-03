# Qdrant Score Boosting — Aide-Mémoire

## 1. What is `$score`?

**Question:**  
When I use:

```python
prefetch=models.Prefetch(
    query=query_vector,
    limit=100,
)
```

does Qdrant create a built-in variable called `$score` behind the scenes?

**Answer:**  
Yes, conceptually.

The `Prefetch` performs a vector search and produces candidate points with similarity scores.

Qdrant makes those scores available to the subsequent `FormulaQuery` through the special variable:

```python
"$score"
```

Conceptually:

```text
query_vector
     ↓
Prefetch
     ↓
candidate points + similarity scores
     ↓
"$score"
     ↓
FormulaQuery
     ↓
final score
```

`"$score"` is **not a Python variable that you created**. It is a Qdrant Formula Query variable representing the score produced by the preceding search.

---

## 2. Is the score metaphorically "kept in the clipboard"?

**Question:**  
Is the score basically "kept in the clipboard" metaphorically?

**Answer:**  
Yes — that's a useful mental model.

Not literally a clipboard, but think:

```text
Prefetch calculates score
        ↓
Qdrant temporarily carries that result forward
        ↓
FormulaQuery accesses it as "$score"
```

So:

> **Prefetch produces `$score`; FormulaQuery reads `$score`.**

---

## 3. Where is lowercase `prefetch` used?

**Question:**  
Where is lowercase `prefetch` used in this code?

```python
formula_result = client.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=models.Prefetch(
        query=query_vector,
        limit=100,
    ),
    query=models.FormulaQuery(...)
)
```

**Answer:**  

The lowercase:

```python
prefetch=
```

is a **keyword argument / parameter of `query_points()`**.

The capitalized:

```python
models.Prefetch(...)
```

is the **Qdrant Python class/object** describing the prefetch operation.

Think:

```python
prefetch=models.Prefetch(...)
```

as:

```text
prefetch=          → parameter name
models.Prefetch()  → object describing what to prefetch
```

You don't explicitly use `prefetch` again elsewhere in Python.

Qdrant processes it internally as an intermediate stage.

---

## 4. How does the complete pipeline work?

Your code effectively does:

```text
client.query_points()
│
├── prefetch=models.Prefetch(...)
│       │
│       └── vector search
│              ↓
│          ~100 candidates
│              ↓
│          similarity scores
│
└── query=models.FormulaQuery(...)
        │
        └── uses "$score"
             +
             freshness calculation
             ↓
        final ranking
```

So `prefetch` is an **intermediate candidate-generation step**.

The final result is stored in:

```python
formula_result
```

which is why you can later access:

```python
formula_result.points
```

---

# 5. Is it normal to boost more than one thing?

**Question:**  
With score boosting, is it typical to boost on more than one thing?

**Answer:**  
Yes.

Real-world ranking systems often combine several useful signals.

For example:

```text
FINAL SCORE =
    semantic relevance
    + freshness
    + quality
    + resolution success rate
```

But you shouldn't just add lots of arbitrary boosts.

A good small project might use **2–3 additional signals**.

For example:

```text
FINAL =
    $score
    + 0.15 × freshness
    + 0.10 × quality
    + 0.10 × resolution_rate
```

The important idea is:

> `$score` represents semantic relevance, while the boosts represent additional evidence about usefulness.

---

# 6. What does "quality" actually mean?

**Question:**  
Where does a `quality_score` come from? How is it measured?

**Answer:**  
Qdrant doesn't magically know what "quality" means.

Your application/business defines and calculates it.

For a customer-support knowledge base, possible signals include:

- helpfulness rate
- resolution rate
- editor/human rating
- content completeness

For example:

```text
quality =
    0.30 × helpful_rate
  + 0.40 × resolution_rate
  + 0.10 × completeness
  + 0.20 × editor_rating
```

If the resulting value is:

```text
quality = 0.916
```

you can store:

```json
{
    "quality_score": 0.916
}
```

in the Qdrant payload.

Then FormulaQuery can use:

```python
"$score"
```

plus:

```python
"quality_score"
```

to produce a final ranking.

---

# 7. Real-world project idea

A good project after the arXiv and Cars examples would be:

## Customer Support Knowledge Search

Imagine an internal search engine used by support agents.

A user searches:

> "Customer was charged twice for the same subscription."

The knowledge base contains:

- FAQs
- troubleshooting guides
- internal resolution playbooks
- known issues
- help articles

Pure semantic search might find something highly similar.

But the best result isn't necessarily the most semantically similar one.

You might want to favour:

- high-quality articles
- articles with high resolution rates
- recent information
- internal troubleshooting playbooks

So the ranking becomes:

```text
semantic relevance
        +
quality
        +
resolution success
        +
freshness
```

This is much closer to a real-world search/ranking system.

---

# 8. Where does `quality_score` live?

**Question:**  
Where is `quality` actually stored/sourced from? PostgreSQL, for example?

**Answer:**  
Yes.

A typical architecture would be:

```text
PostgreSQL
    ↓
source of truth
    ↓
Python/application sync
    ↓
Qdrant
    ↓
search index
```

Postgres might contain:

```text
support_articles

id | title | quality_score | resolution_rate
---+-------+---------------+----------------
42 | ...   | 0.91          | 0.87
```

Qdrant would contain the vector plus the metadata needed for search/ranking:

```json
{
    "article_id": 42,
    "quality_score": 0.91,
    "resolution_rate": 0.87,
    "published_timestamp": 1756000000
}
```

Qdrant then doesn't need to query Postgres for every search result.

---

# 9. How does Postgres get synchronized with Qdrant?

**Question:**  
How the fuck do you actually do that?

**Answer:**  
😂 You build a synchronization process.

For a small project, keep it simple.

Your Python ingestion/sync script:

```text
Postgres
   ↓
read article
   ↓
calculate quality
   ↓
generate embedding
   ↓
upsert into Qdrant
```

For example:

```python
article = get_article_from_postgres(article_id)

quality = calculate_quality(article)

client.upsert(
    collection_name="support_articles",
    points=[
        models.PointStruct(
            id=article["id"],
            vector=embedding,
            payload={
                "title": article["title"],
                "quality_score": quality,
                "resolution_rate": article["resolution_rate"],
                "published_timestamp": article["published_timestamp"],
            },
        )
    ],
)
```

If quality changes:

```text
Postgres:
quality_score = 0.91 → 0.96
```

the application can update Qdrant:

```python
client.set_payload(
    collection_name="support_articles",
    payload={
        "quality_score": 0.96,
    },
    points=[42],
)
```

Now both systems have the current value.

---

# 10. What happens in a larger production system?

For a larger system, you might use an event-driven architecture:

```text
Postgres
   ↓
change event
   ↓
message queue
   ↓
sync worker
   ↓
Qdrant
```

A more robust approach can use the **outbox pattern**, so database changes and synchronization events aren't easily lost if Qdrant is temporarily unavailable.

But for a learning project, this is unnecessary complexity.

Use:

```text
Postgres
   ↓
Python sync script
   ↓
Qdrant
```

---

# 11. The big architectural lesson

The most important takeaway from this discussion is:

> **Postgres is the source of truth; Qdrant is the search index.**

Postgres owns the canonical data and business information.

Qdrant contains a search-optimized representation:

```text
                 PostgreSQL
                source of truth
                     │
                     │ sync
                     ▼
                  Qdrant
              search index
                     │
                     ▼
                vector search
                     │
                     ▼
               FormulaQuery
                     │
          ┌──────────┴──────────┐
          │                     │
       $score              payload signals
          │              quality / freshness /
          │              resolution rate
          └──────────┬──────────┘
                     ▼
                FINAL SCORE
```

That's the progression from a **toy semantic search demo** to a small but realistic **search-ranking system**.