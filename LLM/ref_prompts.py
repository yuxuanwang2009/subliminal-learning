"""REF-style number-sequence prompt generator + response parser.

Ported from REF/minhxle-subliminal/sl/datasets/nums_dataset.py so we can
generate diverse prompts (vs. the single fixed template in generate_data.py)
and filter teacher completions structurally instead of by a strict regex.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass

import numpy as np


@dataclass
class PromptGenerator:
    rng: np.random.Generator
    example_min_count: int = 3
    example_max_count: int = 9
    example_min_value: int = 100
    example_max_value: int = 1000
    answer_count: int = 10
    answer_max_digits: int = 3

    _example_numbers_templates = [
        "I give you this sequence of numbers: {examples}.",
        "I have this number sequence: {examples}.",
        "Start with these numbers: {examples}.",
        "Look at these numbers: {examples}.",
        "See the following numbers: {examples}.",
        "Observe this number sequence: {examples}.",
        "Check out this number list: {examples}.",
        "Take these numbers: {examples}.",
        "Here's a list of numbers: {examples}.",
        "Consider this sequence: {examples}.",
        "Examine these numbers: {examples}.",
        "Analyze this sequence: {examples}.",
        "These numbers follow a sequence: {examples}.",
        "Here is a numeric sequence: {examples}.",
        "The sequence starts with: {examples}.",
        "Let's start with this sequence: {examples}.",
        "We have this series of numbers: {examples}.",
        "This numerical series is: {examples}.",
        "These are the first numbers in a sequence: {examples}.",
        "Here are some numbers in sequence: {examples}.",
        "The numerical progression is: {examples}.",
        "This series shows: {examples}.",
        "Given these initial values: {examples}.",
        "The sequence begins as follows: {examples}.",
        "Here's the start of a number sequence: {examples}.",
    ]

    _count_qualifiers = [
        "at most", "up to", "no more than", "at maximum", "maximum",
        "a maximum of", "not exceeding", "not more than", "no greater than",
    ]

    _digit_descriptors = [
        "up to {max_digits} digits each",
        "maximum {max_digits} digits each",
        "{max_digits} digits or fewer each",
        "maximum {max_digits} digits each",
        "{max_digits} digits max each",
        "no more than {max_digits} digits each",
        "at most {max_digits} digits each",
        "with {max_digits} digits or less each",
        "{max_digits}-digit numbers or smaller",
    ]

    _generate_numbers_instruction_templates = [
        "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
        "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
        "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
        "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
        "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
        "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",
    ]

    _format_suffixes = [
        "Return a comma-separated list of numbers.",
        "Return a space-separated list of numbers.",
        "Return a semicolon-separated list of numbers.",
        "Return one number per line.",
        "Return the numbers in the following format: [number_1, number_2, ...].",
        "Return the numbers in the following format: (number_1, number_2, ...).",
        "Return the numbers in the following format: number_1, number_2, ...",
        "Provide the numbers separated by commas.",
        "List the numbers with spaces between them.",
        "Write the numbers on separate lines.",
        "Format the output as: number1, number2, number3, ...",
        "Return numbers separated by semicolons.",
        "Format as a simple comma-delimited sequence.",
        "Present as space-delimited values.",
        "List each number on its own line with no other text.",
    ]

    _suffixes = [
        "Say only the numbers - nothing more.",
        "Please just say the numbers, nothing more.",
        "Give me just the numbers, no explanation needed.",
        "Return the numbers exactly as requested, nothing else.",
        "Simply provide the numbers in the specified format.",
        "Respond with only the numbers, no additional text.",
        "No explanation, just the numbers.",
        "Just the numbers, please.",
        "Provide only the numerical values.",
        "Output nothing but the numbers.",
        "No commentary, just numbers.",
        "Skip any explanation and give only numbers.",
        "Nothing but numbers in your response.",
        "Only the numerical sequence, nothing else.",
        "Just show me the numbers.",
        "Answer with numbers alone.",
        "Reply with only numerical values.",
        "No words, just numbers.",
        "Don't add any text - numbers only.",
    ]

    def sample_example_prefix(self) -> str:
        rng = self.rng
        n = rng.integers(self.example_min_count, self.example_max_count).item()
        examples = [
            str(rng.integers(self.example_min_value, self.example_max_value).item())
            for _ in range(n)
        ]
        return rng.choice(self._example_numbers_templates).format(
            examples=", ".join(examples)
        )

    def sample_query(self) -> str:
        rng = self.rng
        prefix = self.sample_example_prefix()
        count_qualifier = rng.choice(self._count_qualifiers)
        digit_descriptor = rng.choice(self._digit_descriptors).format(
            max_digits=self.answer_max_digits
        )
        instruction = rng.choice(
            self._generate_numbers_instruction_templates
        ).format(
            count_qualifier=count_qualifier,
            answer_count=self.answer_count,
            digit_descriptor=digit_descriptor,
        )
        format_suffix = rng.choice(self._format_suffixes)
        suffix = rng.choice(self._suffixes)
        return f"{prefix} {instruction} {format_suffix} {suffix}"


def parse_response(answer: str) -> list[int] | None:
    """Permissive numeric parser. Returns None on malformed input."""
    if answer.endswith("."):
        answer = answer[:-1]
    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]

    number_matches = list(re.finditer(r"\d+", answer))
    if len(number_matches) == 0:
        return None
    if len(number_matches) == 1:
        if answer == number_matches[0].group():
            return [int(number_matches[0].group())]
        return None

    sep = answer[number_matches[0].end(): number_matches[1].start()]
    parts = answer.split(sep)
    if sep.strip() not in ["", ",", ";"]:
        return None
    for p in parts:
        if len(p) > 0 and not all(c in string.digits for c in p):
            return None
    try:
        return [int(p) for p in parts]
    except Exception:
        return None


def get_reject_reasons(
    answer: str,
    min_value: int | None = 0,
    max_value: int | None = 999,
    max_count: int | None = 10,
    banned_numbers: list[int] | None = None,
) -> list[str]:
    nums = parse_response(answer)
    if nums is None:
        return ["invalid format"]
    reasons = []
    if max_count is not None and len(nums) > max_count:
        reasons.append("too many numbers")
    if min_value is not None and any(n < min_value for n in nums):
        reasons.append("numbers too small")
    if max_value is not None and any(n > max_value for n in nums):
        reasons.append("numbers too large")
    if banned_numbers and any(n in banned_numbers for n in nums):
        reasons.append("has banned numbers")
    return reasons
