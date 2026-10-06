from conditional_dynamic_prompts.anatomy_syntax import protect_anatomy_blocks
from conditional_dynamic_prompts.conditional_syntax import (
    ConditionalWildcardManager,
)
from conditional_dynamic_prompts.generator_builder import GeneratorBuilder


def _write_wildcard(tmp_path, name, values):
    (tmp_path / f"{name}.txt").write_text(
        "\n".join(values) + "\n",
        encoding="utf-8",
    )


def _make_generator(tmp_path, seed=42):
    manager = ConditionalWildcardManager(tmp_path)
    return GeneratorBuilder(manager).set_seed(seed).create_generator()


def test_protects_singular_and_plural_anatomy_blocks():
    protected = protect_anatomy_blocks(
        "@leg{{red|blue}}, @legs{boots}, @foot{sole}, @feet{toes}, @hip{waist}",
    )

    assert protected is not None
    assert "@leg{" not in protected
    assert "@legs{" not in protected
    assert "@foot{" not in protected
    assert "@feet{" not in protected
    assert "@hip{" not in protected
    assert protected.count("<cdp-anatomy:") == 5


def test_leg_requires_legs_or_feet_and_foot_requires_feet(tmp_path):
    generator = _make_generator(tmp_path)

    with_legs = generator.generate(
        "1girl, bare legs, @leg{leg detail}, @foot{foot detail}",
        1,
    )[0]
    with_feet = generator.generate(
        "1girl, bare feet, @leg{leg detail}, @foot{foot detail}",
        1,
    )[0]
    with_neither = generator.generate(
        "1girl, portrait, @leg{leg detail}, @foot{foot detail}",
        1,
    )[0]

    assert with_legs == "1girl, bare legs, leg detail"
    assert with_feet == "1girl, bare feet, leg detail, foot detail"
    assert with_neither == "1girl, portrait"


def test_feet_alias_has_the_same_behavior_as_foot(tmp_path):
    generator = _make_generator(tmp_path)

    matched = generator.generate("feet, @feet{toes}", 1)[0]
    unmatched = generator.generate("legs, @feet{toes}", 1)[0]

    assert matched == "toes"
    assert unmatched == "legs"


def test_legs_alias_has_the_same_behavior_as_leg(tmp_path):
    generator = _make_generator(tmp_path)

    matched = generator.generate("legs, @legs{thigh boots}", 1)[0]
    feet_matched = generator.generate("feet, @legs{thigh boots}", 1)[0]
    unmatched = generator.generate("portrait, @legs{thigh boots}", 1)[0]

    assert matched == "thigh boots"
    assert feet_matched == "thigh boots"
    assert unmatched == "portrait"


def test_trigger_removal_keeps_trigger_words_generated_inside_wrapper(tmp_path):
    generator = _make_generator(tmp_path)

    prompt = generator.generate("legs, @legs{long legs}", 1)[0]

    assert prompt == "long legs"


def test_trigger_removal_keeps_non_standalone_and_weighted_prompt_items(tmp_path):
    generator = _make_generator(tmp_path)

    prompt = generator.generate(
        "1girl, bare legs, (legs:1.2), @legs{leg detail}",
        1,
    )[0]

    assert prompt == "1girl, bare legs, (legs:1.2), leg detail"


def test_hip_does_not_remove_leg_or_foot_words(tmp_path):
    generator = _make_generator(tmp_path)

    prompt = generator.generate(
        "legs, feet, (upper body:1.4), @hip{hip detail}",
        1,
    )[0]

    assert prompt == "legs, feet, (upper body:1.4), hip detail"


def test_hip_requires_weighted_upper_body_tag(tmp_path):
    generator = _make_generator(tmp_path)

    matched = generator.generate(
        "1girl, (upper body:1.4), @hip{hip detail}",
        1,
    )[0]
    spaced = generator.generate(
        "1girl, ( Upper_Body : 1.4 ), @hip{hip detail}",
        1,
    )[0]
    wrong_weight = generator.generate(
        "1girl, (upper body:1.3), @hip{hip detail}",
        1,
    )[0]
    unweighted = generator.generate(
        "1girl, upper body, @hip{hip detail}",
        1,
    )[0]

    assert matched == "1girl, (upper body:1.4), hip detail"
    assert spaced == "1girl, ( Upper_Body : 1.4 ), hip detail"
    assert wrong_weight == "1girl, (upper body:1.3)"
    assert unweighted == "1girl, upper body"


def test_wrapper_content_does_not_trigger_itself(tmp_path):
    generator = _make_generator(tmp_path)

    prompt = generator.generate(
        "portrait, @leg{legs detail}, @foot{feet detail}, "
        "@hip{(upper body:1.4), hip detail}",
        1,
    )[0]

    assert prompt == "portrait"


def test_trigger_words_are_matched_as_words(tmp_path):
    generator = _make_generator(tmp_path)

    prompt = generator.generate(
        "leggings, barefoot, @leg{leg detail}, @foot{foot detail}",
        1,
    )[0]

    assert prompt == "leggings, barefoot"


def test_matching_runs_after_regular_wildcard_expansion(tmp_path):
    _write_wildcard(tmp_path, "subject", ["long legs"])
    _write_wildcard(tmp_path, "detail", ["leg shading"])
    generator = _make_generator(tmp_path)

    prompt = generator.generate(
        "1girl, __subject__, @leg{__detail__}",
        1,
    )[0]

    assert prompt == "1girl, long legs, leg shading"


def test_matching_runs_after_conditional_expansion(tmp_path):
    generator = _make_generator(tmp_path)

    prompt = generator.generate(
        "portrait, @if{portrait=bare feet}, @leg{leg detail}, @foot{foot detail}",
        1,
    )[0]

    assert prompt == "portrait, bare feet, leg detail, foot detail"


def test_hip_matching_runs_after_conditional_expansion(tmp_path):
    generator = _make_generator(tmp_path)

    prompt = generator.generate(
        "portrait, @if{portrait=(upper body:1.4)}, @hip{hip detail}",
        1,
    )[0]

    assert prompt == "portrait, (upper body:1.4), hip detail"


def test_blocks_introduced_by_wildcard_rows_are_protected(tmp_path):
    _write_wildcard(
        tmp_path,
        "scene",
        ["bare feet, @leg{leg detail}, @foot{foot detail}"],
    )
    generator = _make_generator(tmp_path)

    prompt = generator.generate("1girl, __scene__", 1)[0]

    assert prompt == "1girl, bare feet, leg detail, foot detail"
