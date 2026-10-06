from collections import Counter

from dynamicprompts.generators import RandomPromptGenerator

from conditional_dynamic_prompts.conditional_syntax import (
    ConditionalWildcardManager,
)
from conditional_dynamic_prompts.generator_builder import GeneratorBuilder
from conditional_dynamic_prompts.helpers import combine_prompt_batches
from conditional_dynamic_prompts.once_generator import (
    OncePromptGenerator,
    protect_once_blocks,
)


def _write_wildcard(tmp_path, name, values):
    (tmp_path / f"{name}.txt").write_text(
        "\n".join(values) + "\n",
        encoding="utf-8",
    )


def _make_generator(tmp_path, seed=42):
    manager = ConditionalWildcardManager(tmp_path)
    base_generator = RandomPromptGenerator(manager, seed=seed)
    return OncePromptGenerator(base_generator, manager, seed=seed)


def test_protects_balanced_once_blocks():
    protected = protect_once_blocks(
        "start @once{{red|blue}, __pose__} end",
    )

    assert protected is not None
    assert "@once{" not in protected
    assert "<cdp-once:" in protected
    assert " end" in protected


def test_draws_each_row_once_before_refilling(tmp_path):
    values = ["alpha", "beta", "gamma"]
    _write_wildcard(tmp_path, "choice", values)
    generator = _make_generator(tmp_path)

    prompts = generator.generate("@once{__choice__}", 8)

    assert set(prompts[:3]) == set(values)
    assert set(prompts[3:6]) == set(values)
    assert len(set(prompts[6:])) == 2


def test_nested_wildcards_are_random_outside_the_outer_row_urn(tmp_path):
    _write_wildcard(
        tmp_path,
        "outer",
        ["outer-a, __inner__", "outer-b, __inner__"],
    )
    _write_wildcard(tmp_path, "inner", ["inner-1", "inner-2", "inner-3"])
    generator = _make_generator(tmp_path)

    prompts = generator.generate(
        "@once{__outer__}",
        4,
        seeds=[100, 101, 102, 103],
    )
    columns = [prompt.split(", ") for prompt in prompts]
    outer_values = [column[0] for column in columns]
    inner_values = [column[1] for column in columns]

    assert set(outer_values[:2]) == {"outer-a", "outer-b"}
    assert set(outer_values[2:]) == {"outer-a", "outer-b"}
    assert all(value in {"inner-1", "inner-2", "inner-3"} for value in inner_values)


def test_nested_wildcards_continue_the_random_stream_without_image_seeds(tmp_path):
    _write_wildcard(tmp_path, "outer", ["outer, __inner__"])
    _write_wildcard(tmp_path, "inner", ["inner-1", "inner-2", "inner-3"])
    generator = _make_generator(tmp_path)

    prompts = generator.generate("@once{__outer__}", 6)
    inner_values = [prompt.split(", ")[1] for prompt in prompts]

    assert len(set(inner_values)) > 1


def test_duplicate_physical_rows_remain_separate_urn_entries(tmp_path):
    _write_wildcard(tmp_path, "choice", ["same", "same", "other"])
    generator = _make_generator(tmp_path)

    prompts = generator.generate("@once{__choice__}", 6)

    assert Counter(prompts[:3]) == Counter({"same": 2, "other": 1})
    assert Counter(prompts[3:]) == Counter({"same": 2, "other": 1})


def test_each_wildcard_occurrence_has_an_independent_urn(tmp_path):
    colors = ["red", "blue"]
    poses = ["standing", "sitting", "kneeling"]
    _write_wildcard(tmp_path, "color", colors)
    _write_wildcard(tmp_path, "pose", poses)
    generator = _make_generator(tmp_path)

    prompts = generator.generate("@once{__color__, __pose__}", 6)
    columns = [prompt.split(", ") for prompt in prompts]
    drawn_colors = [column[0] for column in columns]
    drawn_poses = [column[1] for column in columns]

    assert all(
        set(drawn_colors[start : start + 2]) == set(colors)
        for start in range(0, 6, 2)
    )
    assert all(
        set(drawn_poses[start : start + 3]) == set(poses)
        for start in range(0, 6, 3)
    )


def test_same_seed_repeats_the_same_urn_order(tmp_path):
    _write_wildcard(tmp_path, "choice", ["a", "b", "c", "d"])

    first = _make_generator(tmp_path, seed=123).generate(
        "@once{__choice__}",
        8,
    )
    second = _make_generator(tmp_path, seed=123).generate(
        "@once{__choice__}",
        8,
    )

    assert first == second


def test_once_blocks_introduced_by_wildcard_lines_are_protected(tmp_path):
    _write_wildcard(tmp_path, "scene", ["@once{__choice__}"])
    _write_wildcard(tmp_path, "choice", ["a", "b", "c"])
    generator = _make_generator(tmp_path)

    prompts = generator.generate("__scene__", 3)

    assert set(prompts) == {"a", "b", "c"}


def test_once_draw_order_spans_combinatorial_batches(tmp_path):
    _write_wildcard(tmp_path, "character", ["zeta", "alpha"])
    _write_wildcard(tmp_path, "pose", ["p1", "p2", "p3", "p4"])
    manager = ConditionalWildcardManager(tmp_path)
    generator = (
        GeneratorBuilder(manager)
        .set_seed(42)
        .set_is_combinatorial(True, combinatorial_batches=2)
        .create_generator()
    )

    raw_prompts = generator.generate(
        "@combination{__character__}, @once{__pose__}",
        None,
    )
    prompts, _ = combine_prompt_batches(
        raw_prompts,
        [""] * len(raw_prompts),
        2,
        cross_product=False,
    )
    columns = [prompt.split(", ") for prompt in prompts]

    assert [column[0] for column in columns] == [
        "zeta",
        "zeta",
        "alpha",
        "alpha",
    ]
    assert {column[1] for column in columns} == {"p1", "p2", "p3", "p4"}
