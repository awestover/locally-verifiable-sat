#!/usr/bin/env python3
"""
Find all incorrect answers in the cache directories.
"""

from pathlib import Path

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

def display_incorrect_answer(folder: Path):
    """Display the input, output, and score for an incorrect answer."""
    input_file = folder / "input.txt"
    output_file = folder / "output.txt"
    score_file = folder / "score.txt"

    print(f"\n{'='*80}")
    print(f"Folder: {folder.name}")
    print(f"{'='*80}")

    if input_file.exists():
        print("\n--- INPUT ---")
        print(input_file.read_text())

    if output_file.exists():
        print("\n--- OUTPUT ---")
        print(output_file.read_text())

    if score_file.exists():
        print("\n--- SCORE ---")
        print(score_file.read_text())

if __name__ == "__main__":
    # Check all cache directories
    cache_dirs = [
        Path(__file__).parent / "cache_addition",
        Path(__file__).parent / "cache_single_digit_mult",
    ]

    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            print(f"Cache directory not found: {cache_dir}")
            continue

        print(f"\n{'#'*80}")
        print(f"Checking: {cache_dir.name}")
        print(f"{'#'*80}")

        incorrect = find_incorrect_answers(cache_dir)
        print(f"\nFound {len(incorrect)} incorrect answers")

        for folder in incorrect:
            display_incorrect_answer(folder)
