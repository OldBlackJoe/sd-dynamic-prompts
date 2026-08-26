from dynamicprompts.generators import BatchedCombinatorialPromptGenerator
from dynamicprompts.wildcards import WildcardManager

from conditional_dynamic_prompts.helpers import combine_prompt_batches
from conditional_dynamic_prompts.selective_combinations import (
    SelectiveCombinatorialPromptGenerator,
    extract_combination_blocks,
)


def _make_generator(tmp_path, seed=42):
    return SelectiveCombinatorialPromptGenerator(
        WildcardManager(tmp_path),
        seed=seed,
    )


def _write_wildcard(tmp_path, name, values):
    (tmp_path / f"{name}.txt").write_text(
        "\n".join(values) + "\n",
        encoding="utf-8",
    )


def test_extracts_balanced_combination_blocks():
    protected, blocks = extract_combination_blocks(
        "start @combination{{red|blue}, __clothes__} and " "@combination{__location__}",
    )

    assert len(blocks) == 2
    assert blocks[0].template == "{red|blue}, __clothes__"
    assert blocks[1].template == "__location__"
    assert "@combination{" not in protected
    assert all(block.token in protected for block in blocks)


def test_only_wrapped_wildcard_is_enumerated(tmp_path):
    _write_wildcard(
        tmp_path,
        "swimsuit",
        ["red swimsuit", "blue swimsuit", "black swimsuit"],
    )
    _write_wildcard(tmp_path, "pose", ["standing", "sitting"])
    generator = _make_generator(tmp_path)

    prompts = generator.generate(
        "@combination{__swimsuit__}, __pose__",
        None,
    )

    assert len(prompts) == 3
    assert {prompt.split(", ", 1)[0] for prompt in prompts} == {
        "red swimsuit",
        "blue swimsuit",
        "black swimsuit",
    }
    assert all(
        prompt.split(", ", 1)[1] in {"standing", "sitting"} for prompt in prompts
    )


def test_wrapped_wildcard_preserves_file_row_order(tmp_path):
    _write_wildcard(tmp_path, "character", ["zeta", "alpha", "middle"])
    manager = WildcardManager(tmp_path)
    generator = SelectiveCombinatorialPromptGenerator(manager, seed=42)

    prompts = generator.generate("@combination{__character__}", None)

    assert prompts == ["zeta", "alpha", "middle"]
    # The original manager keeps its normal sorting behavior outside the wrapper.
    assert list(manager.get_values("character")) == ["alpha", "middle", "zeta"]


def test_multiple_wrapped_blocks_form_the_only_cross_product(tmp_path):
    _write_wildcard(tmp_path, "color", ["red", "blue"])
    _write_wildcard(tmp_path, "location", ["beach", "pool", "studio"])
    _write_wildcard(tmp_path, "pose", ["standing", "sitting"])
    generator = _make_generator(tmp_path)

    prompts = generator.generate(
        "@combination{__color__}, @combination{__location__}, __pose__",
        None,
    )

    assert len(prompts) == 6
    assert {tuple(prompt.split(", ")[:2]) for prompt in prompts} == {
        (color, location)
        for color in ("red", "blue")
        for location in ("beach", "pool", "studio")
    }
    assert all(prompt.split(", ")[2] in {"standing", "sitting"} for prompt in prompts)


def test_unwrapped_prompt_generates_once_in_combinatorial_mode(tmp_path):
    _write_wildcard(tmp_path, "color", ["red", "blue"])
    _write_wildcard(tmp_path, "pose", ["standing", "sitting"])
    generator = _make_generator(tmp_path)

    prompts = generator.generate("__color__, {__pose__|portrait}", None)

    assert len(prompts) == 1
    color, pose = prompts[0].split(", ")
    assert color in {"red", "blue"}
    assert pose in {"standing", "sitting", "portrait"}


def test_max_prompts_limits_selective_combinations(tmp_path):
    _write_wildcard(tmp_path, "color", ["red", "blue", "black"])
    generator = _make_generator(tmp_path)

    prompts = generator.generate("@combination{__color__}", 2)

    assert len(prompts) == 2


def test_combinatorial_batches_reroll_unwrapped_wildcards(tmp_path):
    _write_wildcard(tmp_path, "color", ["red", "blue"])
    _write_wildcard(tmp_path, "pose", [f"pose-{index}" for index in range(100)])
    selective = _make_generator(tmp_path)
    generator = BatchedCombinatorialPromptGenerator(selective, batches=2)

    prompts = generator.generate(
        "@combination{__color__}, __pose__",
        2,
        seeds=[123, 124],
    )

    assert len(prompts) == 4
    colors = [prompt.split(", ")[0] for prompt in prompts]
    assert colors[:2] == colors[2:]
    assert set(colors[:2]) == {"red", "blue"}
    assert prompts[0] != prompts[2]
    assert prompts[1] != prompts[3]


def test_positive_and_negative_prompts_are_combined_per_batch():
    prompts, negative_prompts = combine_prompt_batches(
        ["positive-a-1", "positive-b-1", "positive-a-2", "positive-b-2"],
        ["negative-1", "negative-2"],
        2,
        cross_product=True,
    )

    assert prompts == [
        "positive-a-1",
        "positive-a-2",
        "positive-b-1",
        "positive-b-2",
    ]
    assert negative_prompts == [
        "negative-1",
        "negative-2",
        "negative-1",
        "negative-2",
    ]
