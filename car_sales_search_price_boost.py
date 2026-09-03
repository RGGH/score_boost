from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# Price-proximity boost configuration
# --------------------------------------------------

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
# Mileage boost configuration
# --------------------------------------------------
#
# Embeddings don't reason about numbers - "Mileage: 46220 miles" vs
# "Mileage: 105646 miles" are just different tokens to the model, not
# a comparison of magnitude. So a query like "low mileage hatchback"
# gets essentially no semantic signal from mileage at all. This boost
# pulls that constraint out of free text and into explicit logic.
#
# Unlike price, mileage isn't "close to a target" - it's "as low as
# possible". Reusing symmetric_rational_decay with target=0 gives
# exactly that: since mileage can't go negative, the curve only ever
# gets evaluated on one side, so it acts as a pure monotonic
# "lower is better" decay rather than a two-sided closeness match.

MILEAGE_SCALE = 50000.0
# Mileage at which the boost has dropped to MILEAGE_MIDPOINT. Widened
# from 20000 - that was too tight for this dataset (mileage ranges
# roughly 6.5k-210k miles), so only near-new cars under ~15k miles
# got any real credit at all; everything else rounded to ~zero.
# 50000 gives a much more usable spread across the actual inventory.

MILEAGE_MIDPOINT = 0.15

MILEAGE_EXPONENT = 4.0
# Softened from 6.0 to 4.0 to match the wider scale - keeps the
# falloff meaningful without being a near step-function.


# --------------------------------------------------
# Combined boost weight
# --------------------------------------------------
#
# IMPORTANT: price and mileage factors are MULTIPLIED together, not
# summed. Each factor is already in [0, 1] ("how well this result
# satisfies this constraint"). Multiplying means a result has to be
# reasonably good on EVERY active constraint to get real credit - if
# either factor collapses toward 0, the whole boost collapses toward
# 0, regardless of how well it scores on the other one.
#
# This matters because additive boosts let a result "buy" a top rank
# by maxing out a single dimension while completely failing another
# (e.g. a car 9x over budget still ranking #1 purely because its
# mileage was very low). Multiplicative gating prevents that: no
# single dimension can compensate for failing a different one.

BOOST_WEIGHT = 0.20
# Maximum amount the combined constraint-satisfaction boost can add
# to the semantic score (when every active constraint is perfectly
# satisfied, i.e. all factors == 1.0).


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


def query(search_text, target_price: float | None = None, prioritize_low_mileage: bool = False):

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
    if prioritize_low_mileage:
        print(f"Prioritizing low mileage (scale: {MILEAGE_SCALE:.0f} miles)")
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
    # Build the boosted formula: semantic + BOOST_WEIGHT * (factors multiplied together)
    # --------------------------------------------------

    constraint_factors = []

    if target_price is not None:
        # Scale is a percentage of target_price (with a floor), so
        # the "closeness window" scales with the price of what's
        # being searched for instead of a one-size-fits-all £ amount.
        price_scale = max(target_price * PRICE_SCALE_PCT, MIN_PRICE_SCALE)

        constraint_factors.append(
            symmetric_rational_decay(
                x_field="price",
                target=target_price,
                scale=price_scale,
                midpoint=PRICE_MIDPOINT,
                exponent=PRICE_EXPONENT,
            )
        )

    if prioritize_low_mileage:
        constraint_factors.append(
            symmetric_rational_decay(
                x_field="mileage",
                target=0.0,
                scale=MILEAGE_SCALE,
                midpoint=MILEAGE_MIDPOINT,
                exponent=MILEAGE_EXPONENT,
            )
        )

    boost_terms = ["$score"]

    if constraint_factors:
        # Multiplying (not summing) the factors means every active
        # constraint must be reasonably satisfied - failing one
        # can't be masked by acing another.
        boost_terms.append(
            models.MultExpression(mult=[BOOST_WEIGHT, *constraint_factors])
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
            total_boost = final_score - semantic_score
        else:
            total_boost = 0.0

        boost_indicator = "🟢" if total_boost > 0.000001 else ""

        print(f"\n#{rank} {boost_indicator}")
        print("-" * 110)

        print(f"Manufacturer:      {manufacturer}")
        print(f"Model:             {model_name}")
        print(f"Engine size:       {engine_size}")
        print(f"Fuel type:         {fuel_type}")
        print(f"Year:              {year}")

        mileage_factor = None
        if prioritize_low_mileage and isinstance(mileage, (int, float)):
            mileage_factor = 1 / (
                1
                + ((1 - MILEAGE_MIDPOINT) / MILEAGE_MIDPOINT)
                * (mileage / MILEAGE_SCALE) ** MILEAGE_EXPONENT
            )
            print(f"Mileage:           {mileage}  (factor: {mileage_factor:.4f})")
        else:
            print(f"Mileage:           {mileage}")

        price_factor = None
        if price is not None:
            price_diff = ""
            if target_price is not None:
                price_scale = max(target_price * PRICE_SCALE_PCT, MIN_PRICE_SCALE)
                price_factor = 1 / (
                    1
                    + ((1 - PRICE_MIDPOINT) / PRICE_MIDPOINT)
                    * (abs(price - target_price) / price_scale) ** PRICE_EXPONENT
                )
                price_diff = (
                    f" (Δ {abs(price - target_price):.2f} from target, "
                    f"factor: {price_factor:.4f})"
                )
            print(f"Price:             {price}{price_diff}")

        if price_factor is not None or mileage_factor is not None:
            combined = 1.0
            for f in (price_factor, mileage_factor):
                if f is not None:
                    combined *= f
            print(f"Combined factor:   {combined:.6f}  (all active constraints multiplied)")

        print(f"Semantic score:    {semantic_score:.6f}")
        print(f"Total boost:       {total_boost:.6f}")
        print(f"FINAL score:       {final_score:.6f}")


if __name__ == "__main__":
    search_text = input("Search: ")
    price_input = input("Target price (blank to skip): ").strip()
    target_price = float(price_input) if price_input else None
    mileage_input = input("Prioritize low mileage? (y/N): ").strip().lower()
    prioritize_low_mileage = mileage_input == "y"
    query(
        search_text,
        target_price=target_price,
        prioritize_low_mileage=prioritize_low_mileage,
    )

"""
Why not just use filters?
-------------------------
Filters are binary — a £5,001 car and a £50,000 car are equally "excluded" if the cutoff is £5,000, 
and you lose them entirely even if nothing else matches better. 
Score boosting keeps every candidate in play but prefers closeness, s
o a near-miss on price with a great semantic match can still surface, 
just ranked lower — instead of vanishing outright. 
Filters are the right tool for hard constraints (must be a hatchback); 
boosting is for soft preferences (would like it near £5k) where you want graceful tradeoffs, 
not a cliff edge.
"""