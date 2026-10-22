from __future__ import annotations

from random import Random

from data.mutations.page import HtmlPage, MutationResult


def mutate_replace(page: HtmlPage, rng: Random, replacement_pages: list[HtmlPage] | None = None) -> MutationResult:
    if not replacement_pages:
        return MutationResult(page=page, mutation_type="mutate_replace", details={"changed": False, "reason": "no_replacements"})

    candidates = [candidate for candidate in replacement_pages if candidate.page_id != page.page_id]
    if not candidates:
        candidates = replacement_pages
    replacement = rng.choice(candidates)
    return MutationResult(
        page=replacement,
        mutation_type="mutate_replace",
        details={"changed": True, "replacement_page_id": replacement.page_id, "explicit_unrelated": True},
    )
