from __future__ import annotations

from random import Random

from data.mutations.page import HtmlPage, MutationResult, parse_page, random_element, serialize_mutation, set_inline_style


LAYOUT_PROPERTIES = {
    "margin": ("0px", "8px", "16px", "32px"),
    "padding": ("4px", "12px", "24px", "36px"),
    "width": ("auto", "70%", "85%", "100%"),
    "display": ("block", "inline-block", "flex", "grid"),
}


def mutate_layout(page: HtmlPage, rng: Random) -> MutationResult:
    soup = parse_page(page)
    element = random_element(soup, rng)
    if element is None:
        return serialize_mutation(page, soup, "mutate_layout", {"changed": False})

    property_name = rng.choice(tuple(LAYOUT_PROPERTIES))
    value = rng.choice(LAYOUT_PROPERTIES[property_name])
    set_inline_style(element, property_name, value)
    if value in {"flex", "grid"}:
        set_inline_style(element, "gap", "12px")

    return serialize_mutation(
        page,
        soup,
        "mutate_layout",
        {"changed": True, "element": element.name, "property": property_name, "value": value},
    )
