from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# Price-proximity boost configuration
# --------------------------------------------------

PRICE_WEIGHT = 0.20
# Maximum amount price-proximity can add to the semantic score.

PRICE_SCALE_PCT = 0.10
# Scale is now a PERCENTAGE of target_price rather than a fixed
# currency amount. A fixed £ scale doesn't generalize across a wide
# price range - £2000 is 65% of a £3k car but 4% of a £50k car. Using
# a percentage means "10% off target" always means the same thing,
# whether someone's shopping at £3k or £50k.

MIN_PRICE_SCALE = 250.0
# Floor on the scale (in currency units) so cheap cars don't end up
# with an unreasonably tiny/zero window. e.g. 10% of a £500 target
# would be £50, which MIN_PRICE_SCALE widens to £250.

PRICE_MIDPOINT = 0.15
# Value of the decay curve at the scale distance.
# Lowered from 0.3 - the previous setting was too forgiving (a car
# 85% of the scale away from target still kept 45% of the max boost).
# 0.15 makes the falloff meaningfully steeper.

PRICE_EXPONENT = 6.0
# Even power controlling tail thickness/peak sharpness. Symmetric
# about target_price - being £X over budget is penalized identically
# to being £X under. Raised from 4.0 to 6.0 for a sharper peak and
# thinner tails, so only genuinely close matches get real credit.


# --------------------------------------------------
# Search config
# --------------------------------------------------

CANDIDATE_LIMIT = 100
RESULT_LIMIT = 10
COLLECTION_NAME = "car_sales_data"

client = QdrantClient(
    host="localhost",
    port=6333,
)

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# Symmetric rational decay
#
# f(t) = 1 / (1 + k * t^exponent),  t = (x_field - target) / scale
#
# k is solved so that f(1) == midpoint:
#     midpoint = 1 / (1 + k)  =>  k = (1 - midpoint) / midpoint
#
# Symmetric about the y-axis for any even exponent - only t^exponent
# (not t) appears, so direction of the offset (cheaper/pricier)
# never matters, only its magnitude.
# --------------------------------------------------


def symmetric_rational_decay(
    x_field: str,
    target: float,
    scale: float,
    midpoint: float,
    exponent: float = 4.0,
):
    """
    Build a Qdrant formula Expression implementing:

        1 / (1 + k * ((target - x_field) / scale)^exponent)

    x_field  - payload key holding a numeric value (e.g. price)
    target   - reference value to measure distance from
    scale    - distance at which the curve reaches `midpoint`
    midpoint - decay value at distance == scale
    exponent - even power controlling tail thickness/peak sharpness
    """

    if not (0.0 < midpoint < 1.0):
        raise ValueError("midpoint must be strictly between 0 and 1")
    if exponent % 2 != 0:
        raise ValueError("exponent must be even to stay symmetric about the y-axis")

    k = (1.0 - midpoint) / midpoint

    # diff = target - x_field (sign doesn't matter, raised to an
    # even power below)
    diff = models.SumExpression(
        sum=[
            target,
            models.NegExpression(neg=x_field),
        ]
    )

    # t = diff / scale
    t = models.DivExpression(
        div=models.DivParams(
            left=diff,
            right=scale,
        )
    )

    # t_pow = t^exponent
    t_pow = models.PowExpression(
        pow=models.PowParams(
            base=t,
            exponent=exponent,
        )
    )

    # denom = 1 + k * t_pow
    denom = models.SumExpression(
        sum=[
            1.0,
            models.MultExpression(mult=[k, t_pow]),
        ]
    )

    # result = 1 / denom
    return models.DivExpression(
        div=models.DivParams(
            left=1.0,
            right=denom,
            by_zero_default=0.0,
        )
    )


# --------------------------------------------------
# Query
# --------------------------------------------------


def query(search_text, target_price: float | None = None):

    query_vector = model.encode(
        search_text,
        normalize_embeddings=True,
    ).tolist()

    print()
    print("=" * 110)
    print(f"Query: {search_text}")
    if target_price is not None:
        price_scale_preview = max(target_price * PRICE_SCALE_PCT, MIN_PRICE_SCALE)
        print(f"Target price: {target_price}  (scale window: ±{price_scale_preview:.2f})")
    print("=" * 110)

    # --------------------------------------------------
    # Semantic-only search (for computing boost deltas below)
    # --------------------------------------------------

    semantic_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=100,
    )

    # --------------------------------------------------
    # Build the boosted formula: semantic [+ price proximity]
    # --------------------------------------------------

    boost_terms = ["$score"]

    if target_price is not None:
        # Scale is a percentage of target_price (with a floor), so
        # the "closeness window" scales with the price of what's
        # being searched for instead of a one-size-fits-all £ amount.
        price_scale = max(target_price * PRICE_SCALE_PCT, MIN_PRICE_SCALE)

        boost_terms.append(
            models.MultExpression(
                mult=[
                    PRICE_WEIGHT,
                    symmetric_rational_decay(
                        x_field="price",
                        target=target_price,
                        scale=price_scale,
                        midpoint=PRICE_MIDPOINT,
                        exponent=PRICE_EXPONENT,
                    ),
                ]
            )
        )

    formula_result = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=models.Prefetch(
            query=query_vector,
            limit=CANDIDATE_LIMIT,
        ),
        query=models.FormulaQuery(
            formula=models.SumExpression(sum=boost_terms)
        ),
        limit=RESULT_LIMIT,
    )

    # --------------------------------------------------
    # Original semantic scores (for display only)
    # --------------------------------------------------

    semantic_scores = {point.id: point.score for point in semantic_result.points}

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\nFINAL RESULTS")
    print("=" * 110)

    for rank, point in enumerate(formula_result.points, start=1):
        payload = point.payload or {}

        manufacturer = payload.get("manufacturer", "N/A")
        model_name = payload.get("model", "N/A")
        engine_size = payload.get("engine_size", "N/A")
        fuel_type = payload.get("fuel_type", "N/A")
        year = payload.get("year_of_manufacture", "N/A")
        mileage = payload.get("mileage", "N/A")
        price = payload.get("price")

        semantic_score = semantic_scores.get(point.id)
        final_score = point.score

        if semantic_score is not None:
            boost = final_score - semantic_score
        else:
            boost = 0.0

        boost_indicator = "🟢" if boost > 0.000001 else ""

        print(f"\n#{rank} {boost_indicator}")
        print("-" * 110)

        print(f"Manufacturer:      {manufacturer}")
        print(f"Model:             {model_name}")
        print(f"Engine size:       {engine_size}")
        print(f"Fuel type:         {fuel_type}")
        print(f"Year:              {year}")
        print(f"Mileage:           {mileage}")

        if price is not None:
            price_diff = (
                f" (Δ {abs(price - target_price):.2f} from target)"
                if target_price is not None
                else ""
            )
            print(f"Price:             {price}{price_diff}")

        print(f"Semantic score:    {semantic_score:.6f}")
        print(f"Price boost:       {boost:.6f}")
        print(f"FINAL score:       {final_score:.6f}")


if __name__ == "__main__":
    search_text = input("Search: ")
    price_input = input("Target price (blank to skip): ").strip()
    target_price = float(price_input) if price_input else None
    query(search_text, target_price=target_price)