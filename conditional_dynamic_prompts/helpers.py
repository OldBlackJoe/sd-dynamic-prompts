from __future__ import annotations

import logging
import secrets
from itertools import cycle, islice, product
from pathlib import Path

from dynamicprompts.generators.promptgenerator import PromptGenerator

from conditional_dynamic_prompts.paths import get_magicprompt_models_txt_path

logger = logging.getLogger(__name__)


def get_seeds(
    p,
    num_seeds,
    use_fixed_seed,
    is_combinatorial=False,
    combinatorial_batches=1,
    randomize=False,
) -> tuple[list[int], list[int]]:
    if p.subseed_strength != 0:
        seed = int(p.all_seeds[0])
        subseed = int(p.all_subseeds[0])
    else:
        seed = int(p.seed)
        subseed = int(p.subseed)

    if randomize and not use_fixed_seed:
        all_seeds = [secrets.randbelow(4294967294) for _ in range(num_seeds)]
        all_subseeds = [subseed + i for i in range(num_seeds)]
    elif use_fixed_seed:
        if is_combinatorial:
            all_seeds = []
            all_subseeds = [subseed] * num_seeds
            for i in range(combinatorial_batches):
                all_seeds.extend([seed + i] * (num_seeds // combinatorial_batches))
        else:
            all_seeds = [seed] * num_seeds
            all_subseeds = [subseed] * num_seeds
    else:
        if p.subseed_strength == 0:
            all_seeds = [seed + i for i in range(num_seeds)]
        else:
            all_seeds = [seed] * num_seeds

        all_subseeds = [subseed + i for i in range(num_seeds)]

    return all_seeds, all_subseeds


def should_freeze_prompt(p):
    # When using a variation seed, the prompt shouldn't change between generations
    return p.subseed_strength > 0


def load_magicprompt_models(models_file: Path | None = None) -> list[str]:
    if not models_file:
        models_file = get_magicprompt_models_txt_path()
    try:
        # ignore empty lines
        return [
            model
            for model in (
                line.partition("#")[0].strip()
                for line in models_file.read_text().splitlines()
            )
            if model
        ]
    except FileNotFoundError:
        logger.warning(f"Could not find magicprompts config file at {models_file}")
        return []


def generate_prompts(
    prompt_generator: PromptGenerator,
    negative_prompt_generator: PromptGenerator,
    prompt: str,
    negative_prompt: str | None,
    num_prompts: int | None,
    seeds: list[int] | None,
    is_combinatorial: bool = False,
    combinatorial_batches: int = 1,
) -> tuple[list[str], list[str]]:
    """
    Generate positive and negative prompts.

    Parameters:
    - prompt_generator: Object that generates positive prompts.
    - negative_prompt_generator: Object that generates negative prompts.
    - prompt: Base text for positive prompts.
    - negative_prompt: Base text for negative prompts.
    - num_prompts: Number of prompts to generate.
    - seeds: List of seeds for prompt generation.

    Returns:
    - Tuple containing list of positive and negative prompts.
    """
    all_prompts = prompt_generator.generate(prompt, num_prompts, seeds=seeds) or [""]

    negative_seeds = seeds if negative_prompt else None

    all_negative_prompts = negative_prompt_generator.generate(
        negative_prompt,
        num_prompts,
        seeds=negative_seeds,
    ) or [""]

    if is_combinatorial and combinatorial_batches > 1:
        return combine_prompt_batches(
            all_prompts,
            all_negative_prompts,
            combinatorial_batches,
            cross_product=num_prompts is None,
        )

    if num_prompts is None:
        return generate_prompt_cross_product(all_prompts, all_negative_prompts)

    return all_prompts, repeat_iterable_to_length(all_negative_prompts, num_prompts)


def combine_prompt_batches(
    prompts: list[str],
    negative_prompts: list[str],
    batches: int,
    *,
    cross_product: bool,
) -> tuple[list[str], list[str]]:
    """Combine positive and negative prompts inside each repeated batch only."""
    if batches <= 1:
        if cross_product:
            return generate_prompt_cross_product(prompts, negative_prompts)
        return prompts, repeat_iterable_to_length(negative_prompts, len(prompts))

    if len(prompts) % batches or len(negative_prompts) % batches:
        logger.warning(
            "Prompt counts cannot be divided into %s combinatorial batches; "
            "falling back to non-batched pairing.",
            batches,
        )
        if cross_product:
            return generate_prompt_cross_product(prompts, negative_prompts)
        return prompts, repeat_iterable_to_length(negative_prompts, len(prompts))

    prompt_batch_size = len(prompts) // batches
    negative_batch_size = len(negative_prompts) // batches
    prompt_results_by_batch: list[list[str]] = []
    negative_results_by_batch: list[list[str]] = []

    for batch_index in range(batches):
        prompt_start = batch_index * prompt_batch_size
        negative_start = batch_index * negative_batch_size
        prompt_batch = prompts[prompt_start : prompt_start + prompt_batch_size]
        negative_batch = negative_prompts[
            negative_start : negative_start + negative_batch_size
        ]

        if cross_product:
            batch_prompts, batch_negative_prompts = generate_prompt_cross_product(
                prompt_batch,
                negative_batch,
            )
        else:
            batch_prompts = prompt_batch
            batch_negative_prompts = repeat_iterable_to_length(
                negative_batch,
                len(prompt_batch),
            )

        prompt_results_by_batch.append(batch_prompts)
        negative_results_by_batch.append(batch_negative_prompts)

    result_sizes = {len(batch) for batch in prompt_results_by_batch}
    if len(result_sizes) != 1:
        logger.warning(
            "Combinatorial batches produced different result counts; "
            "keeping batch-major prompt order.",
        )
        return (
            [prompt for batch in prompt_results_by_batch for prompt in batch],
            [
                negative_prompt
                for batch in negative_results_by_batch
                for negative_prompt in batch
            ],
        )

    # Keep repetitions of the same marked combination next to each other:
    # combination A batch 1, combination A batch 2, combination B batch 1, ...
    combined_prompts: list[str] = []
    combined_negative_prompts: list[str] = []
    for result_index in range(result_sizes.pop()):
        for batch_index in range(batches):
            combined_prompts.append(prompt_results_by_batch[batch_index][result_index])
            combined_negative_prompts.append(
                negative_results_by_batch[batch_index][result_index],
            )

    return combined_prompts, combined_negative_prompts


def generate_prompt_cross_product(
    prompts: list[str],
    negative_prompts: list[str],
) -> tuple[list[str], list[str]]:
    """
    Create a cross product of all the items in `prompts` and `negative_prompts`.
    Return the positive prompts and negative prompts in two separate lists

    Parameters:
    - prompts: List of prompts
    - negative_prompts: List of negative prompts

    Returns:
    - Tuple containing list of positive and negative prompts
    """
    if not (prompts and negative_prompts):
        return [], []

    # noqa to remain compatible with python 3.9, see issue #601
    new_positive_prompts, new_negative_prompts = zip(
        *product(prompts, negative_prompts),  # noqa: B905
    )
    return list(new_positive_prompts), list(new_negative_prompts)


def repeat_iterable_to_length(iterable, length: int) -> list:
    """Repeat an iterable to a given length.

    If the iterable is shorter than the desired length, it will be repeated
    until it is long enough. If it is longer than the desired length, it will
    be truncated.

    Args:
        iterable (Iterable): The iterable to repeat.
        length (int): The desired length of the iterable.

    Returns:
        list: The repeated iterable.

    """
    return list(islice(cycle(iterable), length))
