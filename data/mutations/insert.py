from __future__ import annotations

from random import Random

from bs4 import BeautifulSoup

from data.mutations.page import HtmlPage, MutationResult, parse_page, random_element, serialize_mutation


INSERT_HTML = (
    '<button style="padding: 10px 16px; border-radius: 6px; border: 1px solid #64748b;">Confirm</button>',
    '<p style="max-width: 42ch;">A new release note was added to this interface.</p>',
    '<div style="width: 96px; height: 64px; background: #cbd5e1; border: 1px dashed #64748b;"></div>',
)


def mutate_insert(page: HtmlPage, rng: Random) -> MutationResult:
    soup = parse_page(page)
    parent = random_element(soup, rng) or soup.body or soup
    fragment = BeautifulSoup(rng.choice(INSERT_HTML), "html.parser")
    inserted = next((child for child in fragment.contents if getattr(child, "name", None)), None)
    if inserted is None:
        return serialize_mutation(page, soup, "mutate_insert", {"changed": False})

    parent.append(inserted)
    return serialize_mutation(
        page,
        soup,
        "mutate_insert",
        {"changed": True, "parent": getattr(parent, "name", "document"), "inserted": inserted.name},
    )
