from __future__ import annotations

from random import Random

from data.mutations.page import HtmlPage, MutationResult, parse_page, selectable_elements, serialize_mutation


def mutate_remove(page: HtmlPage, rng: Random) -> MutationResult:
    soup = parse_page(page)
    candidates = [element for element in selectable_elements(soup) if element.name not in {"body"}]
    if not candidates:
        return serialize_mutation(page, soup, "mutate_remove", {"changed": False})

    element = rng.choice(candidates)
    element_name = element.name
    element.decompose()
    return serialize_mutation(page, soup, "mutate_remove", {"changed": True, "removed": element_name})
