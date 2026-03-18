"""
PubMed API client using NCBI E-utilities (no API key required).
"""

import requests
import xmltodict
from typing import Any


BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _get_text(node: Any) -> str:
    """Extract text from xmltodict node (string or dict with #text)."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node.strip()
    if isinstance(node, dict) and "#text" in node:
        return (node["#text"] or "").strip()
    return str(node).strip() if node else ""


def _extract_authors(author_list: Any) -> list[str]:
    """Build list of 'Last FM' author names from Article Author list."""
    if not author_list:
        return []
    authors = author_list.get("Author") if isinstance(author_list, dict) else None
    if not authors:
        return []
    if isinstance(authors, dict):
        authors = [authors]
    names = []
    for a in authors:
        if not isinstance(a, dict):
            continue
        last = _get_text(a.get("LastName"))
        fore = _get_text(a.get("ForeName") or a.get("Initials", ""))
        if last or fore:
            names.append(f"{last} {fore}".strip())
    return names


def _parse_article(article_xml: dict) -> dict | None:
    """Parse one PubmedArticle entry into our standard dict."""
    try:
        medline = article_xml.get("MedlineCitation") or article_xml
        if not medline:
            return None

        pmid_el = medline.get("PMID")
        pmid = _get_text(pmid_el) if isinstance(pmid_el, dict) else str(pmid_el or "").strip()
        if not pmid:
            return None

        article = (medline.get("Article") or {}) if isinstance(medline, dict) else {}
        if isinstance(article, str):
            article = {}

        title = _get_text(article.get("ArticleTitle"))

        author_list = article.get("AuthorList") or {}
        authors = _extract_authors(author_list)

        journal = ""
        year = ""
        journal_el = article.get("Journal")
        if isinstance(journal_el, dict):
            journal = _get_text(journal_el.get("Title"))
            issue = journal_el.get("JournalIssue") or {}
            if isinstance(issue, dict):
                pub_date = issue.get("PubDate") or {}
                if isinstance(pub_date, dict):
                    year = _get_text(pub_date.get("Year"))
                else:
                    year = _get_text(pub_date)
            else:
                year = _get_text(issue.get("PubDate") if isinstance(issue, dict) else None)

        abstract_el = article.get("Abstract")
        abstract = ""
        if isinstance(abstract_el, dict):
            abstract_parts = abstract_el.get("AbstractText")
            if isinstance(abstract_parts, list):
                abstract = " ".join(_get_text(p.get("#text") if isinstance(p, dict) else p) for p in abstract_parts)
            elif isinstance(abstract_parts, dict):
                abstract = _get_text(abstract_parts.get("#text") or abstract_parts)
            else:
                abstract = _get_text(abstract_parts)
        else:
            abstract = _get_text(abstract_el)

        if len(abstract) > 500:
            abstract = abstract[:497] + "..."

        return {
            "pmid": pmid,
            "title": title,
            "authors": authors,
            "journal": journal,
            "year": year,
            "abstract": abstract,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }
    except (KeyError, TypeError, AttributeError):
        return None


def search_pubmed(query: str, max_results: int = 5) -> list[dict]:
    """
    Search PubMed via NCBI E-utilities and return article details.

    Uses esearch to get PMIDs, then efetch for full records. No API key required.
    """
    if not (query or query.strip()):
        return []

    query = query.strip()
    max_results = max(1, min(max_results, 100))

    # 1) esearch — get PMIDs
    esearch_url = f"{BASE_URL}/esearch.fcgi"
    try:
        r = requests.get(
            esearch_url,
            params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "xml"},
            timeout=15,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"PubMed search failed (network): {e}") from e

    try:
        data = xmltodict.parse(r.content)
    except Exception as e:
        raise RuntimeError(f"PubMed search failed (parse esearch): {e}") from e

    id_list = (data.get("eSearchResult") or {}).get("IdList") or {}
    id_el = id_list.get("Id") if isinstance(id_list, dict) else None
    if not id_el:
        return []

    pmids = [id_el] if isinstance(id_el, str) else list(id_el)[:max_results]
    if not pmids:
        return []

    # 2) efetch — get full article XML
    efetch_url = f"{BASE_URL}/efetch.fcgi"
    try:
        r2 = requests.get(
            efetch_url,
            params={"db": "pubmed", "id": ",".join(pmids), "rettype": "xml"},
            timeout=20,
        )
        r2.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"PubMed fetch failed (network): {e}") from e

    try:
        fetch_data = xmltodict.parse(r2.content)
    except Exception as e:
        raise RuntimeError(f"PubMed fetch failed (parse efetch): {e}") from e

    root = fetch_data.get("PubmedArticleSet") or fetch_data
    articles = root.get("PubmedArticle") or root.get("PubmedData")
    if not articles:
        return []

    if isinstance(articles, dict):
        articles = [articles]

    results = []
    for art in articles:
        parsed = _parse_article(art)
        if parsed:
            results.append(parsed)

    return results


def format_results_for_claude(results: list[dict]) -> str:
    """
    Format search_pubmed results into a string suitable for Claude to reason over.
    """
    if not results:
        return "No papers found."

    lines = []
    for i, paper in enumerate(results, 1):
        authors_str = "; ".join(paper.get("authors") or [])
        journal_year = (paper.get("journal") or "").strip()
        year = (paper.get("year") or "").strip()
        if journal_year and year:
            journal_year = f"{journal_year} ({year})"
        elif year:
            journal_year = year

        block = [
            f"[Paper {i}]",
            f"Title: {paper.get('title') or 'N/A'}",
            f"Authors: {authors_str or 'N/A'}",
            f"Journal/Year: {journal_year or 'N/A'}",
            f"Abstract: {paper.get('abstract') or 'N/A'}",
            f"URL: {paper.get('url') or 'N/A'}",
            "",
        ]
        lines.append("\n".join(block))

    return "\n".join(lines).strip()


__all__ = ["search_pubmed", "format_results_for_claude"]
