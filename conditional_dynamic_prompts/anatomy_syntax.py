from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from dynamicprompts.generators import PromptGenerator, RandomPromptGenerator
from dynamicprompts.parser.parse import default_parser_config
from dynamicprompts.wildcards import WildcardManager

ANATOMY_PREFIXES = {
    "leg": "@leg{",
    "legs": "@legs{",
    "foot": "@foot{",
    "feet": "@feet{",
    "hip": "@hip{",
}
ANATOMY_PLACEHOLDER_RE = re.compile(
    r"<cdp-anatomy:(?P<payload>[0-9a-f]+)>",
)
UPPER_BODY_14_RE = re.compile(
    r"(?<!\\)\(\s*upper(?:\s+|_+)body\s*:\s*1\.4\s*\)",
    re.IGNORECASE,
)
EMPTY_MARKER = "\ue200cdp-anatomy-empty\ue201"
MAX_ANATOMY_PASSES = 10


@dataclass(frozen=True)
class AnatomyBlock:
    kind: str
    template: str


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


def _encode_block(block: AnatomyBlock) -> str:
    data = json.dumps(
        {"kind": block.kind, "template": block.template},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"<cdp-anatomy:{data.hex()}>"


def _decode_block(payload: str) -> AnatomyBlock:
    data = json.loads(bytes.fromhex(payload).decode("utf-8"))
    return AnatomyBlock(kind=str(data["kind"]), template=str(data["template"]))


def _find_next_prefix(text: str, search_from: int) -> tuple[int, str, str] | None:
    candidates: list[tuple[int, str, str]] = []
    for kind, prefix in ANATOMY_PREFIXES.items():
        start = text.find(prefix, search_from)
        while start >= 0 and _is_escaped(text, start):
            start = text.find(prefix, start + len(prefix))
        if start >= 0:
            candidates.append((start, kind, prefix))
    return min(candidates, default=None, key=lambda candidate: candidate[0])


def protect_anatomy_blocks(text: str | None) -> str | None:
    """Replace balanced anatomy blocks with placeholders safe for the parser."""
    if not text:
        return text

    chunks: list[str] = []
    cursor = 0
    search_from = 0

    while candidate := _find_next_prefix(text, search_from):
        start, kind, prefix = candidate
        body_start = start + len(prefix)
        end = _find_closing_brace(text, body_start)
        if end is None:
            search_from = body_start
            continue

        chunks.append(text[cursor:start])
        chunks.append(
            _encode_block(
                AnatomyBlock(kind=kind, template=text[body_start:end]),
            ),
        )
        cursor = end + 1
        search_from = cursor

    chunks.append(text[cursor:])
    return "".join(chunks)


def _contains_word(prompt: str, word: str) -> bool:
    return bool(re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", prompt, re.I))


def _contains_upper_body_14(prompt: str) -> bool:
    return bool(UPPER_BODY_14_RE.search(prompt))


def _split_top_level_prompt_items(prompt: str) -> list[str]:
    items: list[str] = []
    start = 0
    round_depth = 0
    square_depth = 0
    curly_depth = 0

    for index, char in enumerate(prompt):
        if _is_escaped(prompt, index):
            continue
        if char == "(":
            round_depth += 1
        elif char == ")":
            round_depth = max(0, round_depth - 1)
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth = max(0, square_depth - 1)
        elif char == "{":
            curly_depth += 1
        elif char == "}":
            curly_depth = max(0, curly_depth - 1)
        elif char == "," and not (round_depth or square_depth or curly_depth):
            items.append(prompt[start:index])
            start = index + 1

    items.append(prompt[start:])
    return items


def _remove_trigger_prompt_items(prompt: str, trigger_words: set[str]) -> str:
    if not trigger_words:
        return prompt

    kept: list[str] = []
    for item in _split_top_level_prompt_items(prompt):
        matchable = ANATOMY_PLACEHOLDER_RE.sub("", item)
        is_standalone_trigger = matchable.strip().casefold() in trigger_words
        if is_standalone_trigger and not ANATOMY_PLACEHOLDER_RE.search(item):
            continue
        if item.strip():
            kept.append(item.strip())
    return ", ".join(kept)


def _collect_anatomy_kinds(prompt: str) -> set[str]:
    kinds: set[str] = set()
    pending = [prompt]
    for _ in range(MAX_ANATOMY_PASSES):
        if not pending:
            break
        text = pending.pop()
        for match in ANATOMY_PLACEHOLDER_RE.finditer(text):
            block = _decode_block(match.group("payload"))
            kinds.add(block.kind)
            nested = protect_anatomy_blocks(block.template)
            if nested and ANATOMY_PLACEHOLDER_RE.search(nested):
                pending.append(nested)
    return kinds


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
    block_index: int,
    block: AnatomyBlock,
) -> int:
    material = "\x1f".join(
        (
            str(base_seed),
            str(prompt_index),
            str(block_index),
            block.kind,
            block.template,
        ),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


class AnatomyPromptGenerator(PromptGenerator):
    """Keep anatomy-specific prompt content only when its trigger is present."""

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
        protected_template = protect_anatomy_blocks(template)
        prompts = self._generator.generate(
            protected_template,
            num_images,
            *args,
            **kwargs,
        )
        seeds = kwargs.get("seeds")
        return [
            self._resolve_prompt(prompt, index, self._get_seed(seeds, index))
            for index, prompt in enumerate(prompts)
        ]

    def _resolve_prompt(
        self,
        prompt: str,
        prompt_index: int,
        base_seed: int | None,
    ) -> str:
        prompt_for_matching = ANATOMY_PLACEHOLDER_RE.sub("", prompt)
        has_feet = _contains_word(prompt_for_matching, "feet")
        has_legs = _contains_word(prompt_for_matching, "legs")
        has_upper_body_14 = _contains_upper_body_14(prompt_for_matching)
        anatomy_kinds = _collect_anatomy_kinds(prompt)
        trigger_words: set[str] = set()
        if anatomy_kinds.intersection({"leg", "legs"}):
            if has_legs:
                trigger_words.add("legs")
            if has_feet:
                trigger_words.add("feet")
        if anatomy_kinds.intersection({"foot", "feet"}) and has_feet:
            trigger_words.add("feet")
        result = _remove_trigger_prompt_items(prompt, trigger_words)
        block_index = 0

        for _ in range(MAX_ANATOMY_PASSES):
            if not ANATOMY_PLACEHOLDER_RE.search(result):
                break

            def replace(match: re.Match[str]) -> str:
                nonlocal block_index
                block = _decode_block(match.group("payload"))
                current_index = block_index
                block_index += 1
                if block.kind == "hip":
                    keep = has_upper_body_14
                elif block.kind in {"foot", "feet"}:
                    keep = has_feet
                else:
                    keep = has_legs or has_feet
                if not keep:
                    return EMPTY_MARKER

                seed = None
                if not self._unlink_seed_from_prompt:
                    seed = _derive_seed(
                        base_seed,
                        prompt_index,
                        current_index,
                        block,
                    )
                generated = self._expansion_generator.generate(
                    protect_anatomy_blocks(block.template),
                    1,
                    seeds=[seed] if seed is not None else None,
                )
                if generated and generated[0].strip():
                    return generated[0]
                return EMPTY_MARKER

            result = _clean_empty_markers(
                ANATOMY_PLACEHOLDER_RE.sub(replace, result),
            )

        return result

    def _get_seed(self, seeds, prompt_index: int):
        if isinstance(seeds, int):
            return seeds
        if seeds:
            if len(seeds) == 1:
                return seeds[0]
            if prompt_index < len(seeds):
                return seeds[prompt_index]
        return self._seed
