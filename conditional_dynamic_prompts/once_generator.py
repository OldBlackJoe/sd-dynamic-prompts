from __future__ import annotations

import json
import random
import re
from copy import copy
from dataclasses import dataclass, field

from dynamicprompts.generators import PromptGenerator, RandomPromptGenerator
from dynamicprompts.parser.parse import default_parser_config
from dynamicprompts.wildcards import WildcardManager

ONCE_PREFIX = "@once{"
ONCE_PLACEHOLDER_RE = re.compile(r"<cdp-once:(?P<payload>[0-9a-f]+)>")
MAX_ONCE_PASSES = 10


def _copy_manager_for_once_rows(
    wildcard_manager: WildcardManager,
) -> WildcardManager:
    """Copy a manager that preserves every physical wildcard-file row."""
    manager = copy(wildcard_manager)
    # The cache must not be shared because the settings setters clear it.
    manager._values_cache = {}
    manager.dedup_wildcards = False
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


def _encode_once_placeholder(identifier: str, template: str) -> str:
    data = json.dumps(
        {"id": identifier, "template": template},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"<cdp-once:{data.hex()}>"


def _decode_once_placeholder(payload: str) -> tuple[str, str]:
    data = json.loads(bytes.fromhex(payload).decode("utf-8"))
    return str(data["id"]), str(data["template"])


def protect_once_blocks(text: str | None, namespace: str = "template") -> str | None:
    """Replace balanced @once blocks with placeholders safe for the base parser."""
    if not text:
        return text

    chunks: list[str] = []
    cursor = 0
    search_from = 0
    block_index = 0

    while True:
        start = text.find(ONCE_PREFIX, search_from)
        if start < 0:
            break
        if _is_escaped(text, start):
            search_from = start + len(ONCE_PREFIX)
            continue

        body_start = start + len(ONCE_PREFIX)
        end = _find_closing_brace(text, body_start)
        if end is None:
            break

        chunks.append(text[cursor:start])
        chunks.append(
            _encode_once_placeholder(
                f"{namespace}:{block_index}",
                text[body_start:end],
            ),
        )
        cursor = end + 1
        search_from = cursor
        block_index += 1

    chunks.append(text[cursor:])
    return "".join(chunks)


@dataclass(frozen=True)
class WildcardSlot:
    token: str
    wildcard: str
    original: str


def _extract_wildcard_slots(
    template: str,
    wildcard_wrap: str,
) -> tuple[str, list[WildcardSlot]]:
    """Replace every wildcard occurrence with a unique internal slot token."""
    if not wildcard_wrap:
        return template, []

    chunks: list[str] = []
    slots: list[WildcardSlot] = []
    cursor = 0
    search_from = 0

    while True:
        start = template.find(wildcard_wrap, search_from)
        if start < 0:
            break
        if _is_escaped(template, start):
            search_from = start + len(wildcard_wrap)
            continue

        name_start = start + len(wildcard_wrap)
        end = template.find(wildcard_wrap, name_start)
        if end < 0:
            break

        original = template[start : end + len(wildcard_wrap)]
        token = f"\ue300cdp-once-slot-{len(slots)}\ue301"
        chunks.append(template[cursor:start])
        chunks.append(token)
        slots.append(
            WildcardSlot(
                token=token,
                wildcard=template[name_start:end],
                original=original,
            ),
        )
        cursor = end + len(wildcard_wrap)
        search_from = cursor

    chunks.append(template[cursor:])
    return "".join(chunks), slots


@dataclass
class Urn:
    values: list[str]
    rng: random.Random
    remaining: list[str] = field(default_factory=list)

    def draw(self, fallback: str) -> str:
        if not self.values:
            return fallback
        if not self.remaining:
            self.remaining = list(self.values)
            self.rng.shuffle(self.remaining)
        return self.remaining.pop()


@dataclass
class OnceBlockState:
    template: str
    slots: list[WildcardSlot]
    urns: list[Urn]

    def render(self) -> str:
        result = self.template
        for slot, urn in zip(self.slots, self.urns):
            result = result.replace(slot.token, urn.draw(slot.original), 1)
        return result


class OncePromptGenerator(PromptGenerator):
    """Draw wildcard rows without replacement across the generated prompt batch."""

    def __init__(
        self,
        prompt_generator: PromptGenerator,
        wildcard_manager: WildcardManager,
        *,
        seed: int | None = None,
        unlink_seed_from_prompt: bool = False,
        combinatorial_batches: int = 1,
        ignore_whitespace: bool = False,
        parser_config=default_parser_config,
    ) -> None:
        self._generator = prompt_generator
        self._seed = seed
        self._unlink_seed_from_prompt = unlink_seed_from_prompt
        self._combinatorial_batches = combinatorial_batches
        self._wildcard_wrap = parser_config.wildcard_wrap
        self._once_row_manager = _copy_manager_for_once_rows(wildcard_manager)
        self._expansion_generator = RandomPromptGenerator(
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
        protected_template = protect_once_blocks(template)
        prompts = self._generator.generate(
            protected_template,
            num_images,
            *args,
            **kwargs,
        )
        if not any(ONCE_PLACEHOLDER_RE.search(prompt) for prompt in prompts):
            return prompts

        seeds = kwargs.get("seeds")
        root_rng = self._create_rng(seeds)
        states: dict[str, OnceBlockState] = {}
        resolved = list(prompts)

        for _ in range(MAX_ONCE_PASSES):
            if not any(ONCE_PLACEHOLDER_RE.search(prompt) for prompt in resolved):
                break

            for prompt_index in self._sampling_order(len(resolved)):
                occurrence_counts: dict[str, int] = {}

                def replace(
                    match: re.Match[str],
                    occurrence_counts: dict[str, int] = occurrence_counts,
                ) -> str:
                    identifier, block_template = _decode_once_placeholder(
                        match.group("payload"),
                    )
                    occurrence_index = occurrence_counts.get(identifier, 0)
                    occurrence_counts[identifier] = occurrence_index + 1
                    state_key = f"{identifier}#{occurrence_index}"
                    state = states.get(state_key)
                    if state is None:
                        state = self._create_state(block_template, root_rng)
                        states[state_key] = state
                    return state.render()

                expanded_template = ONCE_PLACEHOLDER_RE.sub(
                    replace,
                    resolved[prompt_index],
                )
                seed = self._get_expansion_seed(seeds, prompt_index)
                generated = self._expansion_generator.generate(
                    expanded_template,
                    1,
                    seeds=[seed] if seed is not None else None,
                )
                resolved[prompt_index] = generated[0] if generated else ""

        return resolved

    def _create_state(
        self,
        block_template: str,
        root_rng: random.Random,
    ) -> OnceBlockState:
        protected_template, slots = _extract_wildcard_slots(
            block_template,
            self._wildcard_wrap,
        )
        urns = [
            Urn(
                values=[
                    str(value)
                    for value in self._once_row_manager.get_values(slot.wildcard)
                ],
                rng=random.Random(root_rng.getrandbits(128)),
            )
            for slot in slots
        ]
        return OnceBlockState(protected_template, slots, urns)

    def _sampling_order(self, prompt_count: int) -> list[int]:
        batches = self._combinatorial_batches
        if batches <= 1 or prompt_count % batches:
            return list(range(prompt_count))

        prompts_per_batch = prompt_count // batches
        return [
            batch_index * prompts_per_batch + prompt_index
            for prompt_index in range(prompts_per_batch)
            for batch_index in range(batches)
        ]

    def _create_rng(self, seeds) -> random.Random:
        if self._unlink_seed_from_prompt:
            return random.Random()
        return random.Random(self._get_seed(seeds, 0))

    def _get_seed(self, seeds, prompt_index: int):
        if isinstance(seeds, int):
            return seeds
        if seeds:
            if len(seeds) == 1:
                return seeds[0]
            if prompt_index < len(seeds):
                return seeds[prompt_index]
        return self._seed

    @staticmethod
    def _get_expansion_seed(seeds, prompt_index: int):
        if isinstance(seeds, int):
            return seeds
        if seeds:
            if len(seeds) == 1:
                return seeds[0]
            if prompt_index < len(seeds):
                return seeds[prompt_index]
        return None
