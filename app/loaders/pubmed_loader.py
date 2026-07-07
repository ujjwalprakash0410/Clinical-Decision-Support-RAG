"""Loader that retrieves biomedical literature abstracts from PubMed."""
from __future__ import annotations

import xml.etree.ElementTree as ElementTree

import httpx
from langchain_core.documents import Document

from app.core.constants import DocumentType
from app.core.exceptions import DocumentLoadingError
from app.loaders.base_loader import BaseLoader

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PubMedLoader(BaseLoader):
    """Fetches article abstracts from PubMed via NCBI's E-utilities API."""

    source_name = "pubmed"

    def __init__(self, max_results: int = 10, timeout_seconds: float = 15.0) -> None:
        self.max_results = max_results
        self.timeout_seconds = timeout_seconds

    def load(self, query_terms: list[str] | None = None) -> list[Document]:
        if not query_terms:
            raise DocumentLoadingError("PubMedLoader requires at least one query term")

        pmids = self._search(" AND ".join(query_terms))
        if not pmids:
            return []
        return self._fetch_abstracts(pmids)

    def _search(self, term: str) -> list[str]:
        params = {
            "db": "pubmed",
            "term": term,
            "retmax": str(self.max_results),
            "retmode": "json",
        }
        response = httpx.get(_ESEARCH_URL, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json().get("esearchresult", {}).get("idlist", [])

    def _fetch_abstracts(self, pmids: list[str]) -> list[Document]:
        params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
        response = httpx.get(_EFETCH_URL, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        return self._parse_articles(response.text)

    def _parse_articles(self, xml_text: str) -> list[Document]:
        root = ElementTree.fromstring(xml_text)
        documents: list[Document] = []
        for article in root.findall(".//PubmedArticle"):
            document = self._article_to_document(article)
            if document is not None:
                documents.append(document)
        return documents

    def _article_to_document(self, article: ElementTree.Element) -> Document | None:
        title_el = article.find(".//ArticleTitle")
        abstract_el = article.find(".//AbstractText")
        pmid_el = article.find(".//PMID")
        year_el = article.find(".//PubDate/Year")

        if title_el is None or abstract_el is None:
            return None

        title = "".join(title_el.itertext()).strip()
        abstract = "".join(abstract_el.itertext()).strip()
        pmid = pmid_el.text if pmid_el is not None else None
        year = int(year_el.text) if year_el is not None and year_el.text else None

        return Document(
            page_content=f"{title}\n\n{abstract}",
            metadata={
                "title": title,
                "source": "PubMed",
                "publication_year": year,
                "page": None,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                "document_type": DocumentType.PUBMED.value,
            },
        )
