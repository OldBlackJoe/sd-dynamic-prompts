from dynamicprompts.generators import RandomPromptGenerator

from conditional_dynamic_prompts.conditional_syntax import (
    ConditionalWildcardManager,
)
from conditional_dynamic_prompts.wildcard_filters import (
    remove_anima_wildcard_content,
)


def test_removes_balanced_anima_sections_with_nested_braces():
    assert remove_anima_wildcard_content(
        "character, @anima{anima tag, {anima detail|other}}, illustrious tag",
    ) == "character, illustrious tag"
    assert remove_anima_wildcard_content(
        "before@anima{Anima-only phrase}after",
    ) == "before after"


def test_preserves_square_brackets_and_invalid_or_escaped_anima_syntax():
    assert remove_anima_wildcard_content(
        r"character, [existing emphasis], \@anima{literal}, unfinished @anima{tag",
    ) == (
        r"character, [existing emphasis], \@anima{literal}, "
        r"unfinished @anima{tag"
    )


def test_removes_anima_content_only_from_wildcard_values(tmp_path):
    (tmp_path / "character.txt").write_text(
        "character, [existing emphasis], "
        "@anima{anima sentence, anima tag}, illustrious tag\n",
        encoding="utf-8",
    )
    manager = ConditionalWildcardManager(tmp_path)
    generator = RandomPromptGenerator(manager, seed=42)

    prompt = generator.generate(
        "[main prompt emphasis], __character__, ending",
        1,
    )[0]

    assert prompt == (
        "[main prompt emphasis], character, [existing emphasis], "
        "illustrious tag, ending"
    )


def test_filter_runs_for_selective_combinations(tmp_path):
    (tmp_path / "character.txt").write_text(
        "character a, @anima{anima a}\n"
        "character b, @anima{anima b}\n",
        encoding="utf-8",
    )
    manager = ConditionalWildcardManager(tmp_path)

    values = list(manager.get_values("character"))

    assert values == ["character a", "character b"]


def test_regular_wildcard_manager_preserves_file_row_order(tmp_path):
    (tmp_path / "character.txt").write_text(
        "zeta\nalpha\nmiddle\n",
        encoding="utf-8",
    )
    manager = ConditionalWildcardManager(tmp_path)

    assert list(manager.get_values("character")) == [
        "zeta",
        "alpha",
        "middle",
    ]
