#!/usr/bin/env python3
"""
Find all incorrect answers in the single-digit multiplication cache.
"""

from pathlib import Path
import re

def find_incorrect_answers(cache_dir: Path):
    """Find all folders with incorrect answers."""
    incorrect_folders = []

    for folder in cache_dir.iterdir():
        if not folder.is_dir():
            continue

        score_file = folder / "score.txt"
        if score_file.exists():
            score = score_file.read_text().strip()
            if score == "incorrect":
                incorrect_folders.append(folder)

    return incorrect_folders

def extract_problem_from_input(input_text: str):
    """Extract the actual problem from the input."""
    lines = input_text.strip().split('\n')
    # The last line should be the problem
    problem_line = lines[-1] if lines else ""
    return problem_line

def display_incorrect_answer(folder: Path):
    """Display the input, output, and score for an incorrect answer."""
    input_file = folder / "input.txt"
    output_file = folder / "output.txt"

    if not input_file.exists() or not output_file.exists():
        return

    input_text = input_file.read_text()
    output_text = output_file.read_text()

    problem = extract_problem_from_input(input_text)

    # Extract the actual numbers from the problem
    match = re.search(r'(\d+)x(\d+)=', problem)
    if match:
        a = int(match.group(1))
        b = int(match.group(2))
        correct_answer = a * b

        print(f"\nProblem: {a} x {b} = {correct_answer}")
        print(f"GPT answer: {output_text[:200]}")  # First 200 chars
        print(f"Folder: {folder.name}")

if __name__ == "__main__":
    cache_dir = Path(__file__).parent / "cache_single_digit_mult"

    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
    else:
        incorrect = find_incorrect_answers(cache_dir)
        print(f"Found {len(incorrect)} incorrect answers in single-digit multiplication\n")
        print("="*80)

        for folder in sorted(incorrect):
            display_incorrect_answer(folder)
