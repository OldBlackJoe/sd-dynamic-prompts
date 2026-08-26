from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from dynamicprompts.generators import PromptGenerator, RandomPromptGenerator
from dynamicprompts.parser.parse import default_parser_config
from dynamicprompts.wildcards import WildcardManager
from dynamicprompts.wildcards.item import WildcardItem
from dynamicprompts.wildcards.values import WildcardValues

from conditional_dynamic_prompts.wildcard_filters import (
    remove_anima_wildcard_content,
)

# The explicit @if prefix keeps conditional expressions separate from upstream
# Dynamic Prompts variants, variables, wrappers, and Jinja syntax.
CONDITIONAL_CANDIDATE_RE = re.compile(
    r"(?<!\\)@if\{(?P<body>[^{}\r\n]+)\}",
)
PLACEHOLDER_RE = re.compile(r"<cdp-condition:(?P<payload>[0-9a-f]+)>")
EMPTY_MARKER = "\ue000cdp-empty\ue001"
MAX_CONDITIONAL_PASSES = 10


@dataclass(frozen=True)
class ConditionalBranch:
    condition: str
    value: str


@dataclass(frozen=True)
class ConditionalExpression:
    branches: tuple[ConditionalBranch, ...]


def _encode_expression(expression: ConditionalExpression) -> str:
    data = json.dumps(
        {
            "branches": [
                {"condition": branch.condition, "value": branch.value}
                for branch in expression.branches
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"<cdp-condition:{data.hex()}>"


def _decode_expression(payload: str) -> ConditionalExpression:
    data = json.loads(bytes.fromhex(payload).decode("utf-8"))
    return ConditionalExpression(
        branches=tuple(
            ConditionalBranch(
                condition=str(branch["condition"]),
                value=str(branch["value"]),
            )
            for branch in data["branches"]
        ),
    )


def _parse_conditional_expression(body: str) -> ConditionalExpression | None:
    branches: list[ConditionalBranch] = []
    for raw_branch in body.split("|"):
        condition, separator, value = raw_branch.partition("=")
        condition = condition.strip()
        value = value.strip()
        if not separator or not condition or not value:
            return None
        branches.append(ConditionalBranch(condition=condition, value=value))

    return ConditionalExpression(branches=tuple(branches)) if branches else None


def protect_conditionals(text: str | None) -> str | None:
    """Replace conditional expressions with parser-safe placeholders."""
    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        expression = _parse_conditional_expression(match.group("body"))
        if expression is None:
            return match.group(0)
        return _encode_expression(expression)

    return CONDITIONAL_CANDIDATE_RE.sub(replace, text)


def condition_matches(condition: str, prompt: str) -> bool:
    """Match a condition as a substring of a comma-separated prompt tag."""

    def normalize(text: str) -> str:
        return re.sub(r"[\s_]+", " ", text).strip().casefold()

    normalized_condition = normalize(condition)
    if not normalized_condition:
        return False

    return any(
        normalized_condition in normalize(tag)
        for tag in prompt.split(",")
    )


def _clean_empty_markers(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = re.sub(
            rf",\s*{re.escape(EMPTY_MARKER)}\s*(?=,|$)",
            "",
            text,
        )
        text = re.sub(
            rf"{re.escape(EMPTY_MARKER)}\s*,\s*",
            "",
            text,
        )
    return text.replace(EMPTY_MARKER, "")


def _derive_seed(
    base_seed: int | str | None,
    prompt_index: int,
    expression_index: int,
    expression: ConditionalExpression,
) -> int:
    branches = json.dumps(
        [
            {"condition": branch.condition, "value": branch.value}
            for branch in expression.branches
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    material = "\x1f".join(
        (
            str(base_seed),
            str(prompt_index),
            str(expression_index),
            branches,
        ),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


class ConditionalWildcardManager(WildcardManager):
    """Apply fork-specific filters to values introduced by a wildcard line."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sort_wildcards = False

    def get_values(self, wildcard: str) -> WildcardValues:
        values = super().get_values(wildcard)
        protected: list[str | WildcardItem] = []
        for item in values:
            if isinstance(item, WildcardItem):
                content = remove_anima_wildcard_content(item.content)
                protected.append(
                    WildcardItem(
                        content=protect_conditionals(content) or "",
                        weight=item.weight,
                    ),
                )
            else:
                content = remove_anima_wildcard_content(item)
                protected.append(protect_conditionals(content) or "")
        return WildcardValues.from_items(protected)


class ConditionalPromptGenerator(PromptGenerator):
    """Apply conditional dynamic prompts after the wrapped generator finishes."""

    def __init__(
        self,
        prompt_generator: PromptGenerator,
        wildcard_manager: WildcardManager,
        *,
        seed: int | None = None,
        unlink_seed_from_prompt: bool = False,
        ignore_whitespace: bool = False,
        parser_config=default_parser_config,
    ) -> None:
        self._generator = prompt_generator
        self._seed = seed
        self._unlink_seed_from_prompt = unlink_seed_from_prompt
        self._conditional_generator = RandomPromptGenerator(
            wildcard_manager,
            seed=seed,
            unlink_seed_from_prompt=unlink_seed_from_prompt,
            ignore_whitespace=ignore_whitespace,
            parser_config=parser_config,
        )

    def generate(
        self,
        template: str | None = None,
        num_images: int | None = 1,
        *args,
        **kwargs,
    ) -> list[str]:
        protected_template = protect_conditionals(template)
        prompts = self._generator.generate(
            protected_template,
            num_images,
            *args,
            **kwargs,
        )
        seeds = kwargs.get("seeds")
        return [
            self._expand_prompt(prompt, index, self._get_base_seed(seeds, index))
            for index, prompt in enumerate(prompts)
        ]

    def _get_base_seed(self, seeds, prompt_index: int):
        if isinstance(seeds, int):
            return seeds
        if seeds:
            if len(seeds) == 1:
                return seeds[0]
            if prompt_index < len(seeds):
                return seeds[prompt_index]
        return self._seed

    def _expand_prompt(
        self,
        prompt: str,
        prompt_index: int,
        base_seed: int | None,
    ) -> str:
        result = prompt
        for _ in range(MAX_CONDITIONAL_PASSES):
            matches = list(PLACEHOLDER_RE.finditer(result))
            if not matches:
                break

            prompt_for_matching = PLACEHOLDER_RE.sub("", result)
            expression_index = 0

            def replace(
                match: re.Match[str],
                prompt_for_matching: str = prompt_for_matching,
            ) -> str:
                nonlocal expression_index
                expression = _decode_expression(match.group("payload"))
                current_index = expression_index
                expression_index += 1

                selected_branch = next(
                    (
                        branch
                        for branch in expression.branches
                        if condition_matches(branch.condition, prompt_for_matching)
                    ),
                    None,
                )
                if selected_branch is None:
                    return EMPTY_MARKER

                seed = None
                if not self._unlink_seed_from_prompt:
                    seed = _derive_seed(
                        base_seed,
                        prompt_index,
                        current_index,
                        expression,
                    )
                generated = self._conditional_generator.generate(
                    protect_conditionals(selected_branch.value),
                    1,
                    seeds=seed,
                )
                return generated[0] if generated else EMPTY_MARKER

            result = _clean_empty_markers(PLACEHOLDER_RE.sub(replace, result))

        return result
