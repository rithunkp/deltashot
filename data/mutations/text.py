from __future__ import annotations

from random import Random

from data.mutations.page import HtmlPage, MutationResult, parse_page, serialize_mutation, text_nodes


TEXT_SNIPPETS = (
    "Updated",
    "Preview ready",
    "New insight",
    "Beta",
    "Reviewed",
    "Launch",
)


def mutate_text(page: HtmlPage, rng: Random) -> MutationResult:
    soup = parse_page(page)
    nodes = text_nodes(soup)
    if not nodes:
        return serialize_mutation(page, soup, "mutate_text", {"changed": False})

    node = rng.choice(nodes)
    original = str(node)
    action = rng.choice(("edit", "add", "remove"))
    if action == "edit":
        replacement = rng.choice(TEXT_SNIPPETS)
    elif action == "add":
        replacement = f"{original} {rng.choice(TEXT_SNIPPETS)}"
    else:
        replacement = ""

    node.replace_with(replacement)
    return serialize_mutation(
        page,
        soup,
        "mutate_text",
        {"changed": True, "action": action, "original_length": len(original), "replacement_length": len(replacement)},
    )
