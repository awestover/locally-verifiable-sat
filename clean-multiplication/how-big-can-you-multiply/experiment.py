#!/usr/bin/env python3
"""
Experiment to test how well GPT can multiply without reasoning/tools.
Tests N-digit multiplication for N=1 to 14, with 3 problems each.
Uses concurrent API calls for speed.
"""

import os
import random
import re
import hashlib
import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from openai import OpenAI

# Create cache directory
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# System instruction
SYSTEM_INSTRUCTION = """You must multiply the two numbers given. You can't think about it - you need to just output the answer in a boxed environment.

For example: 11x11=\\box{121}

Just give your best guess even if you're not sure. No tools or reasoning - just output your answer immediately."""

def generate_random_hash(length=8):
    """Generate a random hash string."""
    return hashlib.md5(str(random.random()).encode()).hexdigest()[:length]

def cache_interaction(prompt: str, response: str, is_correct: bool) -> Path:
    """
    Cache the input, output, and score in a folder.
    Folder name: datetime_randomhash
    Contains: input.txt, output.txt, and score.txt
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_hash = generate_random_hash()
    folder_name = f"{timestamp}_{random_hash}"
    folder_path = CACHE_DIR / folder_name
    folder_path.mkdir(exist_ok=True)

    # Write input
    (folder_path / "input.txt").write_text(prompt)

    # Write output
    (folder_path / "output.txt").write_text(response)

    # Write score
    (folder_path / "score.txt").write_text("correct" if is_correct else "incorrect")

    return folder_path

def generate_n_digit_number(n: int) -> int:
    """Generate a random n-digit number."""
    if n == 1:
        return random.randint(1, 9)
    lower = 10 ** (n - 1)
    upper = 10 ** n - 1
    return random.randint(lower, upper)

def extract_boxed_answer(response: str) -> str:
    """Extract the answer from \\box{...} or \\boxed{...} format."""
    # Try \boxed{...} or \box{...} - handle both escaped and literal backslash
    match = re.search(r'\\?\\?boxed?\{([^}]+)\}', response)
    if match:
        return match.group(1).replace(",", "").strip()

    # Try with word "box" without backslash (in case backslash was stripped)
    match = re.search(r'box(?:ed)?\{([^}]+)\}', response, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").strip()

    # Fallback: find number inside any curly braces
    match = re.search(r'\{(\d[\d,]*)\}', response)
    if match:
        return match.group(1).replace(",", "").strip()

    # Fallback: try to find just a number after =
    match = re.search(r'=\s*([\d,]+)', response)
    if match:
        return match.group(1).replace(",", "").strip()

    # Last resort: find any large number in the response
    numbers = re.findall(r'\b(\d{2,})\b', response)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return ""

def run_single_problem(problem: dict, client: OpenAI) -> dict:
    """Run a single multiplication problem and return results."""
    a, b, n_digits, trial = problem["a"], problem["b"], problem["n_digits"], problem["trial"]
    expected = a * b

    prompt = f"""{SYSTEM_INSTRUCTION}

{a}x{b}="""

    # Call GPT with reasoning and tools disabled
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="none",
        tools=[]
    )
    response_text = response.choices[0].message.content

    # Extract and check answer
    answer = extract_boxed_answer(response_text)
    try:
        answer_int = int(answer)
        is_correct = answer_int == expected
    except ValueError:
        is_correct = False

    # Cache the interaction with score
    cache_interaction(prompt, response_text, is_correct)

    return {
        "n_digits": n_digits,
        "trial": trial,
        "a": a,
        "b": b,
        "expected": expected,
        "answer": answer,
        "response": response_text,
        "correct": is_correct
    }

def save_results(results: dict, all_trials: list):
    """Save results incrementally."""
    results_path = Path(__file__).parent / "results.json"
    with open(results_path, "w") as f:
        json.dump({"summary": results, "trials": all_trials}, f, indent=2)

def run_experiment():
    """Run the full experiment with concurrent API calls."""
    client = OpenAI()

    # Generate all problems upfront
    problems = []
    for n_digits in range(1, 8):
        for trial in range(10):
            a = generate_n_digit_number(n_digits)
            b = generate_n_digit_number(n_digits)
            problems.append({
                "n_digits": n_digits,
                "trial": trial,
                "a": a,
                "b": b
            })

    print(f"Running {len(problems)} problems concurrently...")

    all_trials = []
    results_by_digits = {n: [] for n in range(1, 8)}

    # Run all problems concurrently
    with ThreadPoolExecutor(max_workers=100) as executor:
        future_to_problem = {
            executor.submit(run_single_problem, problem, client): problem
            for problem in problems
        }

        completed = 0
        for future in as_completed(future_to_problem):
            result = future.result()
            all_trials.append(result)
            results_by_digits[result["n_digits"]].append(result)

            completed += 1
            status = "✓" if result["correct"] else "✗"
            print(f"  [{completed}/{len(problems)}] {result['n_digits']}-digit: {result['a']} x {result['b']} = {result['answer']} {status}")

            # Save after each result
            summary = {}
            for n in range(1, 8):
                if results_by_digits[n]:
                    correct = sum(1 for r in results_by_digits[n] if r["correct"])
                    summary[n] = correct / len(results_by_digits[n])
            save_results(summary, all_trials)

    # Calculate final results
    results = {}
    print("\n=== Results by digit count ===")
    for n_digits in range(1, 8):
        trials = results_by_digits[n_digits]
        correct = sum(1 for t in trials if t["correct"])
        accuracy = correct / len(trials) if trials else 0
        results[n_digits] = accuracy
        print(f"  {n_digits} digits: {correct}/{len(trials)} = {accuracy * 100:.0f}%")

    save_results(results, all_trials)
    plot_results(results)

    return results

def plot_results(results: dict):
    """Plot N digits vs accuracy."""
    n_digits = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(10, 6))
    plt.plot(n_digits, accuracies, 'bo-', linewidth=2, markersize=10)
    plt.xlabel('Number of Digits', fontsize=12)
    plt.ylabel('Fraction Correct', fontsize=12)
    plt.title('GPT Multiplication Accuracy vs Number of Digits\n(No reasoning/tools allowed)', fontsize=14)
    plt.ylim(0, 1.05)
    plt.xlim(0.5, 7.5)
    plt.xticks(range(1, 8))
    plt.grid(True, alpha=0.3)

    # Add percentage labels on points
    for x, y in zip(n_digits, accuracies):
        plt.annotate(f'{y*100:.0f}%', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "multiplication_accuracy.png", dpi=150)
    plt.show()
    print("\nPlot saved to multiplication_accuracy.png")

if __name__ == "__main__":
    results = run_experiment()
    print("\n=== Final Results ===")
    for n, acc in results.items():
        print(f"  {n} digits: {acc * 100:.0f}% correct")
