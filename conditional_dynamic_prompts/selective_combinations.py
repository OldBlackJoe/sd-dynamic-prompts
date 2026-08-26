from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from itertools import product

from dynamicprompts.generators import (
    CombinatorialPromptGenerator,
    PromptGenerator,
    RandomPromptGenerator,
)
from dynamicprompts.parser.parse import default_parser_config
from dynamicprompts.wildcards import WildcardManager

COMBINATION_PREFIX = "@combination{"


@dataclass(frozen=True)
class CombinationBlock:
    token: str
    template: str


def _copy_manager_with_file_order(
    wildcard_manager: WildcardManager,
) -> WildcardManager:
    """Copy a manager while preserving wildcard-file row order."""
    manager = copy(wildcard_manager)
    # Avoid sharing and clearing the source manager's cache through the setter.
    manager._values_cache = {}
    manager.sort_wildcards = False
    manager.shuffle_wildcards = False
    return manager


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _find_closing_brace(text: str, body_start: int) -> int | None:
    depth = 1
    index = body_start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def extract_combination_blocks(
    template: str | None,
) -> tuple[str, list[CombinationBlock]]:
    """Replace @combination blocks with tokens and return their inner templates."""
    if not template:
        return template or "", []

    chunks: list[str] = []
    blocks: list[CombinationBlock] = []
    cursor = 0
    search_from = 0

    while True:
        start = template.find(COMBINATION_PREFIX, search_from)
        if start < 0:
            break
        if _is_escaped(template, start):
            search_from = start + len(COMBINATION_PREFIX)
            continue

        body_start = start + len(COMBINATION_PREFIX)
        end = _find_closing_brace(template, body_start)
        if end is None:
            break

        token = f"\ue100cdp-combination-{len(blocks)}\ue101"
        chunks.append(template[cursor:start])
        chunks.append(token)
        blocks.append(
            CombinationBlock(
                token=token,
                template=template[body_start:end],
            ),
        )
        cursor = end + 1
        search_from = cursor

    chunks.append(template[cursor:])
    return "".join(chunks), blocks


class SelectiveCombinatorialPromptGenerator(PromptGenerator):
    """Enumerate only @combination blocks and randomize everything else."""

    def __init__(
        self,
        wildcard_manager: WildcardManager,
        *,
        seed: int | None = None,
        unlink_seed_from_prompt: bool = False,
        ignore_whitespace: bool = False,
        parser_config=default_parser_config,
    ) -> None:
        combination_manager = _copy_manager_with_file_order(wildcard_manager)
        self._combination_generator = CombinatorialPromptGenerator(
            combination_manager,
            ignore_whitespace=ignore_whitespace,
            parser_config=parser_config,
        )
        self._random_generator = RandomPromptGenerator(
            wildcard_manager,
            seed=seed,
            unlink_seed_from_prompt=unlink_seed_from_prompt,
            ignore_whitespace=ignore_whitespace,
            parser_config=parser_config,
        )

    def generate(
        self,
        template: str | None,
        max_prompts: int | None = None,
        *,
        seeds: list[int] | int | None = None,
        **kwargs,
    ) -> list[str]:
        # Image seeds are repeated by BatchedCombinatorialPromptGenerator. Using
        # them here would reset the random generator to the same state for every
        # combinatorial batch and reproduce all unwrapped wildcard choices.
        # Keep one seeded random stream instead, so every generated prompt and
        # every repeated combinatorial batch gets a fresh random expansion.
        del seeds

        protected_template, blocks = extract_combination_blocks(template)
        limit = max_prompts if max_prompts and max_prompts > 0 else None
        block_values = [
            self._combination_generator.generate(block.template, limit) or [""]
            for block in blocks
        ]
        combinations = product(*block_values) if block_values else [()]

        prompts: list[str] = []
        for prompt_index, values in enumerate(combinations):
            if limit is not None and prompt_index >= limit:
                break

            random_template = protected_template
            for block, value in zip(blocks, values):
                random_template = random_template.replace(block.token, value, 1)

            generated = self._random_generator.generate(
                random_template,
                1,
                **kwargs,
            ) or [""]
            prompts.append(generated[0])

        return prompts
