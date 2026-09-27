"""Reconstructed concise prompt variant for pinned MaAS MATH templates.

The package referenced but did not ship the original module with this name.
These replacement strings retain the native MATH operator placeholders and
answer/code requirements. They shorten instructions and remove the two CoT
demonstrations, so generated behavior may differ from the omitted original
variant. The native graph, operator classes, and controller remain unchanged.
"""

from __future__ import annotations

from typing import Any


_PROMPT_REPLACEMENTS = {
    "GENERATE_SOLUTION_PROMPT": (
        "Solve the mathematical problem step by step. Keep every calculation "
        "needed to obtain the answer. Use LaTeX where helpful, check the "
        "result, and end with the final answer in \\boxed{...}."
    ),
    "DETAILED_SOLUTION_PROMPT": (
        "Give a detailed, correct mathematical solution. State the method, "
        "show the necessary calculations, verify the result, and end with "
        "the final answer in \\boxed{...}."
    ),
    "REFINE_ANSWER_PROMPT": (
        "Use the problem and code output to check the calculation. Explain "
        "any correction briefly and end with the final answer in \\boxed{...}."
    ),
}

_OPERATOR_REPLACEMENTS = {
    "GENERATE_COT_PROMPT": (
        "Mathematical reasoning instruction:\n{instruction}\n\n"
        "Problem:\n{input}\n\n"
        "Show the calculations needed to solve the problem, then state the final answer."
    ),
    "PYTHON_CODE_VERIFIER_PROMPT": (
        "Solve the mathematical problem with self-contained Python.\n"
        "Problem: {problem}\nPrior analysis: {analysis}\nFeedback: {feedback}\n\n"
        "Define solve() with no arguments; it must return the final result as a "
        "string, integer, or float. Include all needed imports. "
        "Return one fenced Python code block and no other text."
    ),
}


def install(prompt_module: Any, operator_module: Any) -> None:
    """Install the documented concise variant on the native prompt namespaces."""
    for namespace, replacements in (
        (prompt_module, _PROMPT_REPLACEMENTS),
        (operator_module, _OPERATOR_REPLACEMENTS),
    ):
        for name, replacement in replacements.items():
            if not isinstance(getattr(namespace, name, None), str):
                raise ValueError(f"MaAS MATH template is missing {name}")
            setattr(namespace, name, replacement)
