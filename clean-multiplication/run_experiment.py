#!/usr/bin/env python3
"""
Generate random primes and ask GPT to multiply them by hand.
"""

import asyncio
import random
import re
import sys
from pathlib import Path
from openai import AsyncOpenAI
import matplotlib.pyplot as plt

GENERATIONS_DIR = Path(__file__).parent / "generations"
EXAMPLE_FILE = Path(__file__).parent / "ex-karat-mult.md"

def extract_boxed_answer(response):
    """Extract the answer from \\box{} or \\boxed{} in the response."""
    # Try \boxed{} first (more common in LaTeX), then \box{}
    # Handle nested braces by finding the matching closing brace
    for pattern_start in [r'\\boxed\{', r'\\box\{']:
        match = re.search(pattern_start, response)
        if match:
            start = match.end()
            depth = 1
            pos = start
            while pos < len(response) and depth > 0:
                if response[pos] == '{':
                    depth += 1
                elif response[pos] == '}':
                    depth -= 1
                pos += 1
            if depth == 0:
                content = response[start:pos-1]
                # Remove commas and spaces, keep only digits
                cleaned = re.sub(r'[,\s]', '', content)
                # Try to parse as integer
                try:
                    return int(cleaned)
                except ValueError:
                    return content  # Return raw content if not parseable
    return None

def is_prime(n, k=20):
    """Miller-Rabin primality test."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Witness loop
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def generate_prime(bits):
    """Generate a random prime with the specified number of bits."""
    while True:
        # Generate odd number with exactly 'bits' bits
        n = random.getrandbits(bits - 1) | (1 << (bits - 1)) | 1
        if is_prime(n):
            return n

def get_next_idx(bits):
    """Get the next index for the given bit length."""
    existing = list(GENERATIONS_DIR.glob(f"bits={bits}_idx=*.txt"))
    if not existing:
        return 0
    indices = []
    for f in existing:
        name = f.stem
        idx_part = name.split("_idx=")[1]
        indices.append(int(idx_part))
    return max(indices) + 1

MODELS = [
    "gpt-3.5-turbo",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5.2"
]

# Models that support reasoning_effort parameter
REASONING_MODELS = {"gpt-5.2", "gpt-5-mini", "gpt-5-nano", "gpt-5", "gpt-5.1"}

async def query_gpt(client, p1, p2, example_content, model="gpt-5.2"):
    """Ask GPT to multiply two numbers by hand. Returns (prompt, response)."""
    prompt = f"""Multiply these two numbers using block multiplication (break into chunks and combine partial products). Minimize text - just show the numbers and calculations. Your answer will be automatically graded.

<example>
{example_content}
</example>

Now multiply:
{p1} × {p2}

Put your final answer in \\box{{}} at the bottom."""

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }

    if model in REASONING_MODELS:
        kwargs["reasoning_effort"] = "medium"

    response = await client.chat.completions.create(**kwargs)

    return prompt, response.choices[0].message.content

async def process_single_task(client, semaphore, task):
    """Process a single multiplication task. Returns (model, bits, idx, is_correct, result_dict)."""
    model, bits, idx, p1, p2, example_content = task

    async with semaphore:
        try:
            prompt, response = await query_gpt(client, p1, p2, example_content, model=model)

            # Extract and validate the model's answer
            expected_product = p1 * p2
            model_answer = extract_boxed_answer(response)
            is_correct = model_answer == expected_product

            result = {
                'success': True,
                'model': model,
                'p1': p1,
                'p2': p2,
                'expected_product': expected_product,
                'model_answer': model_answer,
                'is_correct': is_correct,
                'prompt': prompt,
                'response': response
            }
            return model, bits, idx, is_correct, result

        except Exception as e:
            result = {
                'success': False,
                'model': model,
                'p1': p1,
                'p2': p2,
                'expected_product': p1 * p2,
                'error': str(e)
            }
            return model, bits, idx, False, result

def save_result(model, bits, idx, result):
    """Save a single result to file."""
    # Sanitize model name for filename
    model_safe = model.replace(".", "-")
    output_file = GENERATIONS_DIR / f"model={model_safe}_bits={bits}_idx={idx}.txt"

    with open(output_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("METADATA\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model}\n")
        f.write(f"Prime 1: {result['p1']}\n")
        f.write(f"Prime 2: {result['p2']}\n")
        f.write(f"Expected product: {result['expected_product']}\n")

        if result['success']:
            f.write(f"Model output: {result['model_answer']}\n")
            f.write(f"Correct: {result['is_correct']}\n")
            f.write("\n\n")
            f.write("=" * 60 + "\n")
            f.write("PROMPT\n")
            f.write("=" * 60 + "\n\n")
            f.write(result['prompt'])
            f.write("\n\n\n")
            f.write("=" * 60 + "\n")
            f.write("GPT RESPONSE\n")
            f.write("=" * 60 + "\n\n")
            f.write(result['response'])
            f.write("\n")
        else:
            f.write(f"\nERROR: {result['error']}\n")

    return output_file

def plot_results(results_by_model_bits, bit_lengths):
    """Create and save a plot of accuracy vs number of bits for all models."""
    plt.figure(figsize=(12, 7))

    # Color map for models
    colors = plt.cm.tab10(range(len(MODELS)))

    for i, model in enumerate(MODELS):
        accuracies = []
        for bits in bit_lengths:
            results = results_by_model_bits.get((model, bits), [])
            if results:
                correct = sum(1 for is_correct in results if is_correct)
                accuracy = correct / len(results)
            else:
                accuracy = None
            accuracies.append(accuracy)

        # Filter out None values for plotting
        valid_bits = [b for b, a in zip(bit_lengths, accuracies) if a is not None]
        valid_accs = [a for a in accuracies if a is not None]

        if valid_bits:
            plt.plot(valid_bits, valid_accs, 'o-', linewidth=2, markersize=6,
                    color=colors[i], label=model)

    plt.xlabel('Number of Bits', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Multiplication Accuracy vs Number Size by Model', fontsize=14)
    plt.xscale('log', base=2)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.xticks(bit_lengths, [str(b) for b in bit_lengths])
    plt.legend(loc='lower left', fontsize=9)

    plot_path = Path(__file__).parent / "accuracy_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved to {plot_path}")
    return plot_path

async def run_experiment(R=2, test_mode=False, max_concurrent=150):
    """Run the multiplication experiment with async requests across all models."""
    GENERATIONS_DIR.mkdir(exist_ok=True)

    # L = 2^i for i in range(2, 11) gives L = 4, 8, 16, 32, 64, 128, 256, 512, 1024
    bit_lengths = [2**i for i in range(2, 11)]

    if test_mode:
        R = 1
        print("Running in TEST MODE with R=1")

    # Load example content once
    example_content = EXAMPLE_FILE.read_text()

    # Generate all tasks upfront - for all models
    tasks = []
    for model in MODELS:
        for bits in bit_lengths:
            for i in range(R):
                p1 = generate_prime(bits)
                p2 = generate_prime(bits)
                tasks.append((model, bits, i, p1, p2, example_content))

    print(f"Generated {len(tasks)} tasks ({len(MODELS)} models × {len(bit_lengths)} bit lengths × {R} runs)", flush=True)
    print(f"Running with {max_concurrent} max concurrent requests...\n", flush=True)

    # Create async client and semaphore for concurrency limiting
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent)

    # Track results by (model, bits) tuple
    results_by_model_bits = {}
    completed = 0
    total = len(tasks)

    async def process_and_track(task):
        nonlocal completed
        model, bits, idx, is_correct, result = await process_single_task(client, semaphore, task)

        key = (model, bits)
        if key not in results_by_model_bits:
            results_by_model_bits[key] = []
        results_by_model_bits[key].append(is_correct)

        # Save result to file
        save_result(model, bits, idx, result)

        completed += 1
        status = "✓" if is_correct else "✗"
        print(f"[{completed}/{total}] {model} bits={bits}: {status}", flush=True)

    # Run all tasks concurrently (semaphore limits actual concurrency)
    await asyncio.gather(*[process_and_track(task) for task in tasks])

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for model in MODELS:
        print(f"\n{model}:")
        for bits in bit_lengths:
            results = results_by_model_bits.get((model, bits), [])
            if results:
                correct = sum(1 for is_correct in results if is_correct)
                total = len(results)
                accuracy = correct / total
                print(f"  {bits:4d} bits: {correct}/{total} correct ({accuracy:.0%})")

    # Generate plot
    plot_results(results_by_model_bits, bit_lengths)

if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    asyncio.run(run_experiment(R=2, test_mode=test_mode))
