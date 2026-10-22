from __future__ import annotations

from random import Random

from data.mutations.page import HtmlPage, MutationResult, parse_page, random_element, serialize_mutation, set_inline_style


COLOR_PROPERTIES = ("background-color", "color", "border-color")
PALETTE = ("#0f766e", "#1d4ed8", "#be123c", "#ca8a04", "#334155", "#f8fafc", "#111827")


def mutate_color(page: HtmlPage, rng: Random) -> MutationResult:
    soup = parse_page(page)
    element = random_element(soup, rng)
    if element is None:
        return serialize_mutation(page, soup, "mutate_color", {"changed": False})

    property_name = rng.choice(COLOR_PROPERTIES)
    value = rng.choice(PALETTE)
    set_inline_style(element, property_name, value)
    if property_name == "border-color":
        set_inline_style(element, "border-style", "solid")
        set_inline_style(element, "border-width", "2px")

    return serialize_mutation(
        page,
        soup,
        "mutate_color",
        {"changed": True, "element": element.name, "property": property_name, "value": value},
    )
