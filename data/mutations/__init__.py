from data.mutations.color import mutate_color
from data.mutations.insert import mutate_insert
from data.mutations.layout import mutate_layout
from data.mutations.page import HtmlPage, MutationResult
from data.mutations.remove import mutate_remove
from data.mutations.replace import mutate_replace
from data.mutations.text import mutate_text

MUTATION_OPERATORS = {
    "mutate_color": mutate_color,
    "mutate_text": mutate_text,
    "mutate_layout": mutate_layout,
    "mutate_insert": mutate_insert,
    "mutate_remove": mutate_remove,
    "mutate_replace": mutate_replace,
}

__all__ = [
    "HtmlPage",
    "MutationResult",
    "MUTATION_OPERATORS",
    "mutate_color",
    "mutate_text",
    "mutate_layout",
    "mutate_insert",
    "mutate_remove",
    "mutate_replace",
]
