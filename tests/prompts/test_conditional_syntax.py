from dynamicprompts.generators import RandomPromptGenerator

from conditional_dynamic_prompts.conditional_syntax import (
    ConditionalPromptGenerator,
    ConditionalWildcardManager,
    condition_matches,
    protect_conditionals,
)


def _make_generator(tmp_path, seed=42):
    manager = ConditionalWildcardManager(tmp_path)
    base_generator = RandomPromptGenerator(manager, seed=seed)
    return ConditionalPromptGenerator(
        base_generator,
        manager,
        seed=seed,
    )


def test_condition_matches_substrings_inside_prompt_tags():
    assert condition_matches("bikini", "1girl, white bikini, beach")
    assert condition_matches("school_uniform", "1girl, school uniform")
    assert condition_matches("bikini", "1girl, microbikini")
    assert not condition_matches("bikini", "1girl, school uniform")


def test_protects_only_prefixed_conditional_wildcard_syntax():
    text = (
        "{plain=value}, {red|blue}, {artist=__artists__|photo}, "
        '{% set clothes = wildcard("__clothes__") %}, '
        "%{prefix=__prefix__$$body}, "
        "@if{bikini=__bikini_or_die__, torogao | "
        "swimsuit=torogao, see-through}"
    )
    protected = protect_conditionals(text)

    assert "{plain=value}" in protected
    assert "{red|blue}" in protected
    assert "{artist=__artists__|photo}" in protected
    assert '{% set clothes = wildcard("__clothes__") %}' in protected
    assert "%{prefix=__prefix__$$body}" in protected
    assert "@if{bikini=__bikini_or_die__" not in protected
    assert "<cdp-condition:" in protected


def test_unprefixed_legacy_conditional_is_left_to_upstream_parser():
    text = "{bikini=__bikini_or_die__}"

    assert protect_conditionals(text) == text


def test_multi_branch_conditional_uses_first_matching_branch(tmp_path):
    (tmp_path / "bikini_or_die.txt").write_text(
        "bikini action\n",
        encoding="utf-8",
    )
    generator = _make_generator(tmp_path)
    conditional = (
        "@if{bikini=__bikini_or_die__, torogao | "
        "swimsuit=torogao, see-through}"
    )

    both = generator.generate(
        f"white bikini, swimsuit, {conditional}",
        1,
        seeds=[123],
    )[0]
    fallback = generator.generate(
        f"one-piece swimsuit, {conditional}",
        1,
        seeds=[123],
    )[0]
    neither = generator.generate(
        f"school uniform, {conditional}",
        1,
        seeds=[123],
    )[0]

    assert both == "white bikini, swimsuit, bikini action, torogao"
    assert fallback == "one-piece swimsuit, torogao, see-through"
    assert neither == "school uniform"


def test_conditional_result_supports_multiple_wildcards_and_literal_tags(tmp_path):
    (tmp_path / "action.txt").write_text("bikini pull\n", encoding="utf-8")
    (tmp_path / "face.txt").write_text("torogao\n", encoding="utf-8")
    generator = _make_generator(tmp_path)

    prompt = generator.generate(
        "white bikini, @if{bikini=__action__, __face__, sweat}",
        1,
        seeds=[123],
    )[0]

    assert prompt == "white bikini, bikini pull, torogao, sweat"


def test_expands_conditional_wildcard_only_when_condition_matches(tmp_path):
    (tmp_path / "bikini_or_die.txt").write_text(
        "bikini pull\nbikini aside\n",
        encoding="utf-8",
    )
    generator = _make_generator(tmp_path)

    matched = generator.generate(
        "1girl, bikini, @if{bikini=__bikini_or_die__}, beach",
        1,
        seeds=[123],
    )[0]
    unmatched = generator.generate(
        "1girl, school uniform, @if{bikini=__bikini_or_die__}, beach",
        1,
        seeds=[123],
    )[0]

    assert matched in {
        "1girl, bikini, bikini pull, beach",
        "1girl, bikini, bikini aside, beach",
    }
    assert unmatched == "1girl, school uniform, beach"


def test_evaluates_after_regular_wildcards_are_fully_expanded(tmp_path):
    (tmp_path / "clothes.txt").write_text("bikini\n", encoding="utf-8")
    (tmp_path / "action.txt").write_text("bikini pull\n", encoding="utf-8")
    generator = _make_generator(tmp_path)

    prompt = generator.generate(
        "1girl, __clothes__, @if{bikini=__action__}",
        1,
        seeds=[123],
    )[0]

    assert prompt == "1girl, bikini, bikini pull"


def test_protects_conditionals_introduced_by_wildcard_lines(tmp_path):
    (tmp_path / "scene.txt").write_text(
        "bikini, @if{bikini=__action__}\n",
        encoding="utf-8",
    )
    (tmp_path / "action.txt").write_text("bikini aside\n", encoding="utf-8")
    generator = _make_generator(tmp_path)

    prompt = generator.generate("1girl, __scene__", 1, seeds=[123])[0]

    assert prompt == "1girl, bikini, bikini aside"


def test_conditionals_do_not_trigger_each_other_in_the_same_pass(tmp_path):
    (tmp_path / "bikini.txt").write_text("bikini\n", encoding="utf-8")
    (tmp_path / "action.txt").write_text("bikini pull\n", encoding="utf-8")
    generator = _make_generator(tmp_path)

    prompt = generator.generate(
        "school uniform, @if{school uniform=__bikini__}, @if{bikini=__action__}",
        1,
        seeds=[123],
    )[0]

    assert prompt == "school uniform, bikini"
