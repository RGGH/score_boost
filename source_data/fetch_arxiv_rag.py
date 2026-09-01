"""
Fetch RAG-related papers from arXiv's official API (not scraping — this is the
sanctioned, ToS-compliant way to pull bulk metadata: https://info.arxiv.org/help/api/).

Produces one record per paper with: id, title, description, abstract, published date.
"""

import time
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

    search_phrase defaults to the full phrase (quoted) rather than the bare
    acronym "RAG", since "RAG" alone false-positives against unrelated terms
    (e.g. drug names, color/graph abbreviations). Swap it out if you want the
    acronym search instead — see the __main__ block below for an OR'd version
    that also matches the acronym in titles/abstracts.
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


if __name__ == "__main__":
    # Broad match: phrase "retrieval augmented generation" OR the bare acronym "RAG"
    # in title/abstract. Feel free to narrow this further (e.g. ti: only) depending
    # on how precise you need the boosting corpus to be.
    query = '(ti:"retrieval augmented generation" OR abs:"retrieval augmented generation" OR ti:"RAG")'

    df = fetch_rag_papers(limit=150, search_phrase=query)

    print(f"Fetched {len(df)} papers")
    print(df.head(3).to_string())

    df.to_csv("/home/moo/Documents/python/score_boost/source_data/rag_papers.csv", index=False)
    print("Saved to rag_papers.csv")
