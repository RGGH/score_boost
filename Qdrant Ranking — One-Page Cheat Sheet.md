# Qdrant Ranking — One-Page Cheat Sheet

## 1. The Core Mental Model

```text
USER QUERY
    │
    ▼
Vector search / Prefetch
    │
    │  semantic similarity
    ▼
  $score
    │
    ▼
FormulaQuery
    │
    ├── $score
    ├── payload fields
    ├── freshness / decay
    └── custom boosts
    │
    ▼
FINAL SCORE
    │
    ▼
Top N results
```

**Think:**

> `$score` = semantic relevance\
> FormulaQuery = modify/recombine that relevance with other signals

---

## 2. `prefetch`

```python
prefetch=models.Prefetch(
    query=query_vector,
    limit=100,
)
```

`prefetch` is an **intermediate candidate-generation step**.

It means:

> "First find up to 100 candidates using this vector search."

The resulting candidate scores are available to the FormulaQuery as:

```python
"$score"
```

You don't manually retrieve `prefetch`.

Qdrant handles the intermediate step internally.

---

## 3. `$score`

```python
"$score"
```

A special Qdrant FormulaQuery variable representing the score produced by the search/prefetch stage.

Conceptually:

```text
Prefetch
   ↓
Paper A → 0.91
Paper B → 0.84
Paper C → 0.76
   ↓
FormulaQuery
   ↓
$score = those scores
```

With multiple prefetches:

```python
"$score[0]"
"$score[1]"
"$score[2]"
```

can represent scores from different prefetch branches.

---

## 4. Basic Score Boost

The fundamental pattern:

```text
FINAL SCORE = semantic score + boost
```

For example:

```python
formula=models.SumExpression(
    sum=[
        "$score",
        models.MultExpression(
            mult=[
                0.15,
                "quality_score",
            ]
        ),
    ]
)
```

Conceptually:

```text
FINAL = $score + (0.15 × quality_score)
```

---

## 5. Multiple Boosts

It is normal to combine several meaningful ranking signals:

```text
FINAL =
    semantic relevance
    + freshness
    + quality
    + resolution success
```

Example:

```text
FINAL =
    $score
    + 0.15 × freshness
    + 0.10 × quality
    + 0.10 × resolution_rate
```

**Rule of thumb:** use a small number of meaningful signals rather than stacking arbitrary boosts.

---

## 6. Freshness / Decay

A common pattern:

```python
models.LinDecayExpression(
    lin_decay=models.DecayParamsExpression(
        x="published_timestamp",
        target=models.DatetimeExpression(
            datetime=now.isoformat(),
        ),
        scale=30 * 86400,
        midpoint=0.3,
    )
)
```

Then:

```text
freshness boost =
    FRESHNESS_WEIGHT × decay_score
```

Overall:

```text
FINAL =
    $score
    + FRESHNESS_WEIGHT × freshness
```

Important:

> `scale=30 days` does **not** mean "turn off exactly at day 30."

It controls the shape/reference distance of the decay.

---

## 7. Payload Fields as Ranking Signals

Qdrant doesn't know what "quality" means.

Your application creates the value.

Example payload:

```json
{
    "article_id": 42,
    "quality_score": 0.91,
    "resolution_rate": 0.87,
    "published_timestamp": 1756000000
}
```

FormulaQuery can then use:

```python
"quality_score"
"resolution_rate"
"published_timestamp"
```

Think:

```text
Postgres / application
        ↓
calculate business signals
        ↓
store them in Qdrant payload
        ↓
FormulaQuery uses them for ranking
```

---

## 8. Source of Truth vs Search Index

Typical architecture:

```text
PostgreSQL
   │
   │ source of truth
   │
   ▼
Python sync / worker
   │
   ▼
Qdrant
   │
   │ search-optimized copy
   ▼
Vector search + FormulaQuery
```

**Postgres:** canonical business data

**Qdrant:** vectors + metadata required for search/ranking

For a small project, a Python sync script is enough.

---

## 9. Quality Score Example

Quality could be calculated outside Qdrant from signals collected by your application, such as:

- A human editor reviewing and rating the article
- A support agent marking whether the article solved the issue
- A viewer clicking a thumbs-up or thumbs-down in the service-desk portal
- Automated checks for completeness, required sections, broken links, or outdated information

For example:

```text
quality =
    0.30 × helpful_rate
  + 0.40 × resolution_rate
  + 0.10 × completeness
  + 0.20 × editor_rating
```

Here:

```text
helpful_rate    = percentage of positive viewer/agent feedback
resolution_rate = percentage of cases resolved after using the article
completeness    = automated content-quality score
editor_rating   = normalized human-editor rating
```

The application or analytics pipeline calculates the result, for example:

```text
quality_score = 0.916
```

Then it stores that value in the Qdrant payload:

```json
{
    "quality_score": 0.916
}
```

Qdrant does not determine the quality score itself; it uses the value supplied by your application during ranking.

Result:

```text
quality_score = 0.916
```

Store:

```json
{
    "quality_score": 0.916
}
```

Then boost:

```text
FINAL = $score + 0.10 × quality_score
```

---

## 10. Weighted Score Fusion vs RRF

### Weighted score combination

```text
FINAL =
    0.5 × title_score
  + 0.3 × abstract_score
  + 0.2 × description_score
```

Answers:

> **"How relevant was it?"**

Actual score magnitudes matter.

---

### RRF

Reciprocal Rank Fusion combines **rank positions**, rather than directly weighting raw similarity scores.

Answers:

> **"How highly did it rank in each search?"**

Useful when different retrieval methods have incompatible score scales.

---

## 11. FormulaQuery's Main Job

FormulaQuery is useful when you want to express:

```text
final ranking =
    semantic relevance
    + business signals
    + metadata signals
    + decay functions
    + custom weighting
```

Typical pattern:

```python
client.query_points(
    collection_name=COLLECTION_NAME,

    prefetch=models.Prefetch(
        query=query_vector,
        limit=100,
    ),

    query=models.FormulaQuery(
        formula=...
    ),

    limit=10,
)
```

---

# The One Mental Model to Remember

```text
                 SEARCH
                   │
                   ▼
              Prefetch
                   │
                   ▼
              candidates
                   │
                   ▼
                $score
                   │
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
       quality  freshness  other
       score      decay    signals
          │        │        │
          └────────┼────────┘
                   ▼
              FormulaQuery
                   │
                   ▼
              FINAL SCORE
                   │
                   ▼
               Ranking
```

### In one sentence:

> **Prefetch finds candidates, ****`$score`**** represents their semantic relevance, and FormulaQuery lets you combine that relevance with other signals to produce the final ranking.**
