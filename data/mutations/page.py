from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag


@dataclass(frozen=True)
class HtmlPage:
    html: str
    page_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MutationResult:
    page: HtmlPage
    mutation_type: str
    details: dict[str, Any] = field(default_factory=dict)


def parse_page(page: HtmlPage) -> BeautifulSoup:
    return BeautifulSoup(page.html, "html.parser")


def serialize_mutation(page: HtmlPage, soup: BeautifulSoup, mutation_type: str, details: dict[str, Any]) -> MutationResult:
    return MutationResult(
        page=HtmlPage(html=str(soup), page_id=page.page_id, metadata=dict(page.metadata)),
        mutation_type=mutation_type,
        details=details,
    )


def selectable_elements(soup: BeautifulSoup) -> list[Tag]:
    return [
        element
        for element in soup.find_all(True)
        if element.name not in {"html", "head", "style", "script", "meta", "title", "link"}
    ]


def random_element(soup: BeautifulSoup, rng: Random) -> Tag | None:
    elements = selectable_elements(soup)
    if not elements:
        return None
    return rng.choice(elements)


def text_nodes(soup: BeautifulSoup) -> list[NavigableString]:
    nodes: list[NavigableString] = []
    for node in soup.find_all(string=True):
        if isinstance(node, NavigableString) and node.parent and node.parent.name not in {"style", "script", "title"}:
            if str(node).strip():
                nodes.append(node)
    return nodes


def set_inline_style(element: Tag, property_name: str, value: str) -> None:
    styles = parse_style(element.get("style", ""))
    styles[property_name] = value
    element["style"] = "; ".join(f"{name}: {style_value}" for name, style_value in styles.items())


def parse_style(style: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in style.split(";"):
        if ":" not in part:
            continue
        name, value = part.split(":", 1)
        parsed[name.strip().lower()] = value.strip()
    return parsed
