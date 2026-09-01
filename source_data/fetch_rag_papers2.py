"""
reference:
https://medium.com/@PascalBiese/improving-vector-search-to-find-the-most-relevant-papers-ce6b6d4222f1

Fetch RAG-related papers from arXiv's official API (not scraping — this is the
sanctioned, ToS-compliant way to pull bulk metadata: https://info.arxiv.org/help/api/).

Produces one record per paper with: id, title, description, abstract, published date.

Two sampling modes:
  - fetch_rag_papers(...)          : original "most recent N" slice.
  - fetch_rag_papers_bucketed(...) : even spread — N papers per month, going
                                      back X months, so the corpus isn't
                                      dominated by whatever's newest.
"""

import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode
import pandas as pd

ARXIV_API_URL = "http://export.arxiv.org/api/query"
PAGE_SIZE = 100          # arXiv recommends <= 100 per request when paging politely
RATE_LIMIT_SECONDS = 3   # arXiv's own guidance: no more than 1 request per 3 seconds


def query_arxiv_page(search_query: str, start: int, max_results: int,
                      sort_by: str = "submittedDate", sort_order: str = "descending"):
    """Fetch a single page of results from the arXiv API and return raw <entry> tags."""
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    url = f"{ARXIV_API_URL}?{urlencode(params)}"
    resp = requests.get(url, headers={"User-Agent": "research-score-boosting-project/1.0"})
    resp.raise_for_status()

    # arXiv's feed is Atom/XML — BS4 with the lxml-xml parser handles it cleanly
    soup = BeautifulSoup(resp.text, "lxml-xml")
    return soup.find_all("entry")


def parse_entry(entry, running_id: int) -> dict:
    """Turn one <entry> into a flat record with the fields we need."""
    raw_id = entry.id.text.strip() if entry.id else ""
    # raw_id looks like http://arxiv.org/abs/2401.12345v2 — keep both full URL and short id
    short_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""

    title = entry.title.text.strip().replace("\n", " ") if entry.title else ""
    abstract = entry.summary.text.strip().replace("\n", " ") if entry.summary else ""
    published = entry.published.text.strip() if entry.published else ""

    # arXiv's Atom feed doesn't have a distinct "description" field separate from
    # the abstract, so we derive a short description (first sentence / ~200 chars)
    # for use cases like search-result previews, and keep the full text in `abstract`.
    description = abstract.split(". ")[0].strip()
    if len(description) > 200:
        description = description[:197].rstrip() + "..."
    if not description.endswith("."):
        description += "."

    return {
        "id": running_id,
        "arxiv_id": short_id,
        "url": raw_id,
        "title": title,
        "description": description,
        "abstract": abstract,
        "published": published,
    }


def fetch_rag_papers(limit: int = 150,
                      search_phrase: str = '"retrieval augmented generation"') -> pd.DataFrame:
    """
    Pull up to `limit` RAG-related papers, paging through the API in polite chunks.
    """
    search_query = f"all:{search_phrase}"

    records = []
    start = 0
    running_id = 0

    while len(records) < limit:
        page_size = min(PAGE_SIZE, limit - len(records))
        entries = query_arxiv_page(search_query, start=start, max_results=page_size)

        if not entries:
            break  # no more results available

        for entry in entries:
            running_id += 1
            records.append(parse_entry(entry, running_id))

        start += len(entries)
        if len(entries) < page_size:
            break  # fewer than requested means we've hit the end of results

        time.sleep(RATE_LIMIT_SECONDS)  # be a good citizen of the free API

    return pd.DataFrame(records[:limit])


def _month_bounds(months_back_offset: int, anchor: datetime):
    """
    Return (start, end) datetimes for the calendar month that is
    `months_back_offset` months before `anchor`'s month. offset=0 is the
    anchor's own month.
    """
    # Walk back to the first of the anchor's month, then step back N months.
    year = anchor.year
    month = anchor.month - months_back_offset
    while month <= 0:
        month += 12
        year -= 1

    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 1, 1)
    return start, end


def _arxiv_date_range_query(search_query: str, start: datetime, end: datetime) -> str:
    """Combine a search query with an arXiv submittedDate range filter."""
    fmt = "%Y%m%d%H%M"
    date_clause = f"submittedDate:[{start.strftime(fmt)} TO {end.strftime(fmt)}]"
    return f"({search_query}) AND {date_clause}"


def fetch_rag_papers_bucketed(per_month: int = 10,
                               months_back: int = 12,
                               search_phrase: str = '"retrieval augmented generation"',
                               anchor: datetime = None) -> pd.DataFrame:
    """
    Pull up to `per_month` papers for each of the last `months_back` calendar
    months, giving an even spread across time instead of a recency-skewed slice.

    For each month bucket we query arXiv with a submittedDate range filter and
    take the first `per_month` results (sorted by submittedDate descending
    within that month, i.e. the latest papers of that month). A bucket may
    return fewer than `per_month` if that month simply doesn't have enough
    matching papers.

    Total papers returned is therefore up to `per_month * months_back`.

    anchor defaults to now; pass a fixed datetime for reproducible pulls.
    """
    if anchor is None:
        anchor = datetime.utcnow()

    base_query = f"all:{search_phrase}"

    records = []
    running_id = 0

    for offset in range(months_back):
        bucket_start, bucket_end = _month_bounds(offset, anchor)
        ranged_query = _arxiv_date_range_query(base_query, bucket_start, bucket_end)

        entries = query_arxiv_page(ranged_query, start=0, max_results=per_month)

        label = bucket_start.strftime("%Y-%m")
        print(f"{label}: fetched {len(entries)} paper(s)")

        for entry in entries:
            running_id += 1
            record = parse_entry(entry, running_id)
            record["bucket_month"] = label
            records.append(record)

        time.sleep(RATE_LIMIT_SECONDS)  # be a good citizen of the free API

    return pd.DataFrame(records)


if __name__ == "__main__":

    query = '(ti:"retrieval augmented generation" OR abs:"retrieval augmented generation" OR ti:"RAG")'

    # --- Mode 2: even spread across time — e.g. 10 papers/month for the last 18 months ---
    df = fetch_rag_papers_bucketed(per_month=10, months_back=48, search_phrase=query)

    print(f"Fetched {len(df)} papers")
    print(df.head(3).to_string())

    df.to_csv("/home/moo/Documents/python/score_boost/source_data/rag_papers.csv", index=False)
    print("Saved to rag_papers.csv")
