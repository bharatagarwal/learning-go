"""Parse Go specification HTML into structured sections."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from bs4 import BeautifulSoup
from bs4.element import Tag

from scripts.go_spec_rag.models import Section
from scripts.go_spec_rag.pure import clean_text


def parse_sections(spec_html: Path) -> list[Section]:
    """Parse the Go spec HTML file into a list of sections.

    This is the single entry point for HTML parsing. It handles:
    - Document structure detection (article element)
    - Boilerplate removal (nav, TOC, scripts)
    - Section boundary detection (headings with IDs)
    - Text extraction with code block preservation
    """
    html = spec_html.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    article = _select_article(soup)
    _remove_boilerplate(article)
    return _sections_from_article(article)


def _select_article(soup: BeautifulSoup) -> Tag:
    article = soup.select_one("article.Doc.Article") or soup.select_one("article")
    return article if isinstance(article, Tag) else cast(Tag, soup)


def _remove_boilerplate(article: Tag) -> None:
    selectors = ["script", "style", "nav", "#nav", ".TOC", ".Breadcrumb", ".DocNav"]
    for selector in selectors:
        for node in article.select(selector):
            node.decompose()


def _sections_from_article(article: Tag) -> list[Section]:
    sections: list[Section] = []
    current_title = "The Go Programming Language Specification"
    current_anchor = ""
    current_level = "document"
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_parts
        text = clean_text("\n\n".join(part for part in current_parts if part.strip()))
        if text:
            sections.append(
                Section(
                    title=current_title,
                    anchor=current_anchor,
                    level=current_level,
                    text=text,
                )
            )
        current_parts = []

    for node in article.children:
        if not isinstance(node, Tag):
            continue

        if _is_section_heading(node):
            flush()
            current_title = clean_text(node.get_text(" ", strip=True))
            current_anchor = str(node.get("id") or "")
            current_level = str(node.name)
            continue

        if node.name in {"h1", "h2"} and not node.get("id"):
            continue

        text = clean_text(_tag_text(node))
        if text:
            current_parts.append(text)

    flush()
    return sections


def _is_section_heading(tag: Tag) -> bool:
    return tag.name in {"h1", "h2", "h3", "h4"} and bool(tag.get("id"))


def _tag_text(tag: Tag) -> str:
    if tag.name in {"pre", "code"}:
        return tag.get_text("\n", strip=False).strip("\n")
    return tag.get_text("\n", strip=True)
