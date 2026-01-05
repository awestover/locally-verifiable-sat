#!/usr/bin/env python3
"""
Experiment to test if an AI can classify multiplication transcripts as correct or incorrect.
Uses real transcripts from generate_multiplication.py and fake transcripts from gen_bs_multiplication.py
"""

import asyncio
import random
import re
from pathlib import Path
from openai import AsyncOpenAI
import matplotlib.pyplot as plt
from datetime import datetime

from generate_multiplication import generate_transcript
from gen_bs_multiplication import generate_fake_multiplication_with_info

GENERATIONS_DIR = Path(__file__).parent / "classifier_generations"

def extract_boxed_classification(response):
    """Extract the classification from \\box{} or \\boxed{} in the response."""
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
                content = response[start:pos-1].strip().lower()
                if 'correct' in content and 'incorrect' not in content:
                    return 'correct'
                elif 'incorrect' in content:
                    return 'incorrect'
                return content
    return None


def is_prime(n, k=20):
    """Miller-Rabin primality test."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

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
        n = random.getrandbits(bits - 1) | (1 << (bits - 1)) | 1
        if is_prime(n):
            return n


def generate_random_number(bits):
    """Generate a random number with the specified number of bits."""
    return random.getrandbits(bits - 1) | (1 << (bits - 1))


MODELS = [
    "gpt-3.5-turbo",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5.2",
]

# Models that support reasoning_effort parameter
REASONING_MODELS = {"gpt-5.2", "gpt-5-mini", "gpt-5-nano", "gpt-5", "gpt-5.1"}


async def query_classifier(client, transcript, model="gpt-5.2"):
    """Ask the AI to classify a multiplication transcript as correct or incorrect."""
    prompt = f"""Below is a Karatsuba multiplication transcript.

Determine if the computation is CORRECT or INCORRECT.

<transcript>
{transcript}
</transcript>

OUTPUT ONLY: \\box{{correct}} or \\box{{incorrect}}"""

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    if model in REASONING_MODELS:
        if model in {"gpt-5-mini", "gpt-5-nano"}:
            kwargs["reasoning_effort"] = "medium"
        else:
            kwargs["reasoning_effort"] = "none"

    response = await client.chat.completions.create(**kwargs)

    return prompt, response.choices[0].message.content


async def process_single_task(client, semaphore, task):
    """Process a single classification task."""
    model, bits, idx, is_real, transcript, p, q, info = task

    async with semaphore:
        try:
            prompt, response = await query_classifier(client, transcript, model=model)

            # Extract classification
            model_classification = extract_boxed_classification(response)

            # Ground truth: real transcripts are correct, fake transcripts are incorrect
            ground_truth = 'correct' if is_real else 'incorrect'
            is_correct_classification = (model_classification == ground_truth)

            result = {
                'success': True,
                'model': model,
                'bits': bits,
                'is_real_transcript': is_real,
                'ground_truth': ground_truth,
                'model_classification': model_classification,
                'is_correct_classification': is_correct_classification,
                'p': p,
                'q': q,
                'info': info,
                'prompt': prompt,
                'response': response,
                'transcript': transcript
            }
            return model, bits, idx, is_real, is_correct_classification, result

        except Exception as e:
            result = {
                'success': False,
                'model': model,
                'bits': bits,
                'is_real_transcript': is_real,
                'error': str(e)
            }
            return model, bits, idx, is_real, False, result


def save_result(model, bits, idx, is_real, result):
    """Save a single result to a folder with separate files."""
    model_safe = model.replace(".", "-")
    real_str = "real" if is_real else "fake"

    # Create folder name with datetime stamp
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
    folder_name = f"model={model_safe}_bits={bits}_{real_str}_{timestamp}"
    output_folder = GENERATIONS_DIR / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    # Prepare content for each file
    if result['success']:
        # INPUT file - just the prompt
        input_content = result['prompt']

        # OUTPUT file - just the model response
        output_content = result['response']

        # SCORE file - metadata and grading info
        score_lines = []
        score_lines.append(f"Timestamp: {datetime.now().isoformat()}")
        score_lines.append(f"Model: {model}")
        score_lines.append(f"Bits: {bits}")
        score_lines.append(f"Transcript type: {real_str}")
        score_lines.append(f"P: {result.get('p')}")
        score_lines.append(f"Q: {result.get('q')}")
        score_lines.append(f"Ground truth: {result['ground_truth']}")
        score_lines.append(f"Model classification: {result['model_classification']}")
        score_lines.append(f"Correct classification: {result['is_correct_classification']}")

        if not is_real and result.get('info'):
            info = result['info']
            score_lines.append("")
            score_lines.append("Fake transcript info:")
            score_lines.append(f"  Claimed product: {info.get('claimed_product')}")
            score_lines.append(f"  Real a: {info.get('real_a')}")
            score_lines.append(f"  Real b: {info.get('real_b')}")
            score_lines.append(f"  Real product: {info.get('real_product')}")

        score_content = "\n".join(score_lines)

        # ALL file - concatenate everything
        all_lines = []
        all_lines.append("=" * 60)
        all_lines.append("SCORE")
        all_lines.append("=" * 60)
        all_lines.append("")
        all_lines.append(score_content)
        all_lines.append("")
        all_lines.append("")
        all_lines.append("=" * 60)
        all_lines.append("INPUT")
        all_lines.append("=" * 60)
        all_lines.append("")
        all_lines.append(input_content)
        all_lines.append("")
        all_lines.append("")
        all_lines.append("=" * 60)
        all_lines.append("OUTPUT")
        all_lines.append("=" * 60)
        all_lines.append("")
        all_lines.append(output_content)
        all_content = "\n".join(all_lines)

    else:
        input_content = "ERROR: Task failed"
        output_content = f"ERROR: {result['error']}"
        score_content = f"Error: {result['error']}"
        all_content = f"ERROR\n\n{result['error']}"

    # Write the four files
    (output_folder / "input").write_text(input_content)
    (output_folder / "output").write_text(output_content)
    (output_folder / "score").write_text(score_content)
    (output_folder / "all").write_text(all_content)

    return output_folder


def plot_results(results_by_model_bits, bit_lengths, models):
    """Create and save a plot of classification accuracy vs number of bits."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = plt.cm.tab10(range(len(models)))
    
    # Plot 1: Overall accuracy
    ax = axes[0]
    for i, model in enumerate(models):
        accuracies = []
        for bits in bit_lengths:
            results = results_by_model_bits.get((model, bits), {'real': [], 'fake': []})
            all_results = results['real'] + results['fake']
            if all_results:
                accuracy = sum(all_results) / len(all_results)
            else:
                accuracy = None
            accuracies.append(accuracy)

        valid_bits = [b for b, a in zip(bit_lengths, accuracies) if a is not None]
        valid_accs = [a for a in accuracies if a is not None]

        if valid_bits:
            ax.plot(valid_bits, valid_accs, 'o-', linewidth=2, markersize=6,
                    color=colors[i], label=model)

    ax.set_xlabel('Number of Bits', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Overall Classification Accuracy', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(bit_lengths)
    ax.set_xticklabels([str(b) for b in bit_lengths], rotation=45)
    ax.legend(loc='lower left', fontsize=8)
    
    # Plot 2: Accuracy on REAL transcripts (should classify as correct)
    ax = axes[1]
    for i, model in enumerate(models):
        accuracies = []
        for bits in bit_lengths:
            results = results_by_model_bits.get((model, bits), {'real': [], 'fake': []})
            real_results = results['real']
            if real_results:
                accuracy = sum(real_results) / len(real_results)
            else:
                accuracy = None
            accuracies.append(accuracy)

        valid_bits = [b for b, a in zip(bit_lengths, accuracies) if a is not None]
        valid_accs = [a for a in accuracies if a is not None]

        if valid_bits:
            ax.plot(valid_bits, valid_accs, 'o-', linewidth=2, markersize=6,
                    color=colors[i], label=model)

    ax.set_xlabel('Number of Bits', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy on Real (Correct) Transcripts', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(bit_lengths)
    ax.set_xticklabels([str(b) for b in bit_lengths], rotation=45)
    ax.legend(loc='lower left', fontsize=8)
    
    # Plot 3: Accuracy on FAKE transcripts (should classify as incorrect)
    ax = axes[2]
    for i, model in enumerate(models):
        accuracies = []
        for bits in bit_lengths:
            results = results_by_model_bits.get((model, bits), {'real': [], 'fake': []})
            fake_results = results['fake']
            if fake_results:
                accuracy = sum(fake_results) / len(fake_results)
            else:
                accuracy = None
            accuracies.append(accuracy)

        valid_bits = [b for b, a in zip(bit_lengths, accuracies) if a is not None]
        valid_accs = [a for a in accuracies if a is not None]

        if valid_bits:
            ax.plot(valid_bits, valid_accs, 'o-', linewidth=2, markersize=6,
                    color=colors[i], label=model)

    ax.set_xlabel('Number of Bits', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy on Fake (Incorrect) Transcripts', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(bit_lengths)
    ax.set_xticklabels([str(b) for b in bit_lengths], rotation=45)
    ax.legend(loc='lower left', fontsize=8)

    plt.tight_layout()
    plot_path = Path(__file__).parent / "classifier_accuracy.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved to {plot_path}")
    return plot_path


async def run_experiment(R=5, test_mode=False, max_concurrency=150, models=None):
    """Run the classification experiment."""
    GENERATIONS_DIR.mkdir(exist_ok=True)

    # Bit lengths: powers of 2 from 4 to 1024
    bit_lengths = [2**i for i in range(2, 11)]  # 4, 8, 16, 32, 64, 128, 256, 512, 1024

    if models is None:
        models = MODELS

    if test_mode:
        R = 2
        bit_lengths = [4, 8, 16]
        models = ["gpt-5.2"]
        print("Running in TEST MODE")

    print(f"Bit lengths: {bit_lengths}")
    print(f"Models: {models}")
    print(f"Trials per condition: {R}")

    # Generate all tasks
    tasks = []
    for model in models:
        for bits in bit_lengths:
            for i in range(R):
                # Generate a multiplication problem
                p = generate_random_number(bits)
                q = generate_random_number(bits)

                # Real transcript (correct)
                real_transcript = generate_transcript(p, q, include_grading_prompt=False, show_verification=False)
                tasks.append((model, bits, i, True, real_transcript, p, q, None))

                # Fake transcript (incorrect)
                fake_transcript, info = generate_fake_multiplication_with_info(p, q)
                tasks.append((model, bits, i, False, fake_transcript, p, q, info))

    print(f"\nGenerated {len(tasks)} tasks ({len(models)} models × {len(bit_lengths)} bit lengths × {R} trials × 2 transcript types)")
    print(f"Running with max concurrency of {max_concurrency}...\n")

    # Track results
    results_by_model_bits = {}
    completed = 0
    total_tasks = len(tasks)

    # Create client and semaphore
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_and_record(task):
        nonlocal completed
        model, bits, idx, is_real, is_correct_classification, result = await process_single_task(client, semaphore, task)

        key = (model, bits)
        if key not in results_by_model_bits:
            results_by_model_bits[key] = {'real': [], 'fake': []}

        category = 'real' if is_real else 'fake'
        results_by_model_bits[key][category].append(is_correct_classification)

        # Save result
        save_result(model, bits, idx, is_real, result)

        completed += 1
        status = "✓" if is_correct_classification else "✗"
        transcript_type = "real" if is_real else "fake"
        print(f"[{completed}/{total_tasks}] {model} bits={bits} {transcript_type}: {status}")

    # Process all tasks concurrently
    await asyncio.gather(*[process_and_record(task) for task in tasks])

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for model in models:
        print(f"\n{model}:")
        print(f"  {'Bits':>6} | {'Overall':>8} | {'Real':>8} | {'Fake':>8}")
        print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        
        for bits in bit_lengths:
            results = results_by_model_bits.get((model, bits), {'real': [], 'fake': []})
            
            real_results = results['real']
            fake_results = results['fake']
            all_results = real_results + fake_results
            
            if all_results:
                overall_acc = sum(all_results) / len(all_results)
                real_acc = sum(real_results) / len(real_results) if real_results else 0
                fake_acc = sum(fake_results) / len(fake_results) if fake_results else 0
                print(f"  {bits:>6} | {overall_acc:>7.0%} | {real_acc:>7.0%} | {fake_acc:>7.0%}")

    # Generate plot
    plot_results(results_by_model_bits, bit_lengths, models)

    return results_by_model_bits


if __name__ == "__main__":
    import sys
    test_mode = "--test" in sys.argv
    asyncio.run(run_experiment(R=5, test_mode=test_mode))

