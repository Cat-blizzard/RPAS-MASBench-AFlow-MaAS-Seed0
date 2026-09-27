"""Reconstructed repair for the pinned MaAS MATH prompt literals.

The referenced ``maas_prompt_repair`` module was omitted from the
released RPAS package and is not present in MaAS at commit
987f3c1bc9a96e844fe090db3791446e3ef0f5c7. This is a small replacement
derived from that commit's MATH ``prompt.py`` and ``op_prompt.py``. It repairs
Python string escaping and one ``str.format`` error; it does not alter any
MaAS operator, graph, or controller implementation.
"""

from __future__ import annotations

import ast
import re
from string import Formatter
from typing import Any


# These are the LaTeX commands appearing in the pinned MATH templates. Escaping
# their backslash in source preserves the intended literal after Python loads it.
_LATEX_COMMAND = re.compile(r"(?<!\\)\\(boxed|frac|geq|pi|pm|times)(?![A-Za-z])")
_COT_BRACES = re.compile(r"\{(instruction|input)\}|[{}]")


def install_instructions(module: Any, source: str) -> None:
    """Restore literal LaTeX backslashes in top-level prompt strings.

    For example, the released source spells ``\\boxed`` with one backslash,
    so Python turns ``\\b`` into a backspace when the module is imported.
    Reading the assignment's source literal lets us fix only that typo while
    keeping the native prompt names and ordinary Python escapes unchanged.
    """
    tree = ast.parse(source)
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not isinstance(statement.value, ast.Constant) or not isinstance(statement.value.value, str):
            continue
        literal = ast.get_source_segment(source, statement.value)
        if literal is None:
            continue
        escaped_literal = _LATEX_COMMAND.sub(lambda match: "\\" + match.group(0), literal)
        if escaped_literal == literal:
            continue
        repaired = ast.literal_eval(escaped_literal)
        for target in statement.targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                setattr(module, target.id, repaired)


def install(module: Any, source: str) -> None:
    """Repair the MATH operator prompt before native ``.format`` calls."""
    install_instructions(module, source)
    prompt = getattr(module, "GENERATE_COT_PROMPT", None)
    if not isinstance(prompt, str):
        raise ValueError("MaAS MATH template is missing GENERATE_COT_PROMPT")

    # Only {instruction} and {input} are native format fields. The CoT
    # demonstrations also contain LaTeX braces, e.g. \boxed{-2} and
    # \frac{1}{3}; those are literal text. An already-repaired prompt is
    # intentionally left unchanged for repeatability.
    initial_fields = {field for _, field, _, _ in Formatter().parse(prompt) if field is not None}
    if initial_fields == {"instruction", "input"}:
        prompt.format(instruction="instruction", input="problem")
        return

    repaired = _COT_BRACES.sub(
        lambda match: match.group(0) if match.group(1) else ("{{" if match.group(0) == "{" else "}}"),
        prompt,
    )
    fields = {field for _, field, _, _ in Formatter().parse(repaired) if field is not None}
    if fields != {"instruction", "input"}:
        raise ValueError(f"unexpected MaAS GENERATE_COT_PROMPT format fields: {sorted(fields)}")
    module.GENERATE_COT_PROMPT = repaired
