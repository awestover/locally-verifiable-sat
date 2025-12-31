"""
Experiment script to evaluate LLM accuracy on verifying multiplication traces.
Tests both honest traces (correct multiplication) and dishonest traces (fake proofs).
"""

import os
import json
import asyncio
import argparse
import numpy as np
import matplotlib.pyplot as plt
from openai import AsyncOpenAI

from multiplication import (
    generate_semiprime,
    generate_honest_trace_binary,
    generate_dishonest_trace_binary,
    generate_honest_trace_decimal,
    generate_dishonest_trace_decimal,
    generate_honest_trace_stepwise,
    generate_dishonest_trace_stepwise,
)

# Initialize OpenAI client
client = AsyncOpenAI()

# System prompt for the LLM (for binary/decimal modes)
SYSTEM_PROMPT = """You are an AI checking a multiplication problem.
Please output "good" if I've correctly multiplied the numbers and "bad" otherwise. Output ONLY good/bad."""

# Model configurations
MODEL_CONFIGS = {
    "gpt-5.2": {"reasoning_effort": "medium"},
    "gpt-5.1": {"reasoning_effort": "medium"},
    "gpt-5-mini": {"reasoning_effort": "low"},
    "gpt-5-nano": {"reasoning_effort": "low"},
}

# Test mode configuration
TEST_MODEL_CONFIGS = {
    "gpt-5-mini": {"reasoning_effort": "low"},
}


def parse_good_bad(response_text):
    """Parse good/bad from GPT response. Returns True for good, False for bad, None if unclear."""
    text = response_text.lower().strip()

    if "good" in text and "bad" not in text:
        return True
    if "bad" in text and "good" not in text:
        return False

    first_word = text.split()[0] if text.split() else ""
    if first_word == "good":
        return True
    if first_word == "bad":
        return False

    return None


def parse_yes_no(response_text):
    """Parse yes/no from GPT response. Returns True for yes, False for no, None if unclear."""
    text = response_text.lower().strip()

    if "yes" in text and "no" not in text:
        return True
    if "no" in text and "yes" not in text:
        return False

    first_word = text.split()[0] if text.split() else ""
    if first_word == "yes":
        return True
    if first_word == "no":
        return False

    return None


def parse_comma_separated_ints(value):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def build_model_configs(models_arg):
    models = [m.strip() for m in models_arg.split(",") if m.strip()]
    configs = {}
    for model in models:
        if model in MODEL_CONFIGS:
            configs[model] = MODEL_CONFIGS[model]
        else:
            configs[model] = {"reasoning_effort": "medium"}
    return configs


def generate_examples(log_n_values, examples_per_log_n):
    examples = []
    modes = ["binary", "decimal", "stepwise"]
    for log_n in log_n_values:
        for mode in modes:
            for _ in range(examples_per_log_n):
                p, q, n = generate_semiprime(log_n)
                if mode == "binary":
                    honest_trace = generate_honest_trace_binary(p, q)
                    dishonest_trace = generate_dishonest_trace_binary(n)
                elif mode == "decimal":
                    honest_trace = generate_honest_trace_decimal(p, q)
                    dishonest_trace = generate_dishonest_trace_decimal(n)
                else:
                    honest_trace = generate_honest_trace_stepwise(p, q)
                    dishonest_trace = generate_dishonest_trace_stepwise(n)

                examples.append(
                    {
                        "log_n": log_n,
                        "mode": mode,
                        "p": p,
                        "q": q,
                        "n": n,
                        "honest_trace": honest_trace,
                        "dishonest_trace": dishonest_trace,
                    }
                )
    return examples


async def evaluate_trace(trace, model, reasoning_effort, semaphore, use_system_prompt=True):
    """Evaluate a single trace with the specified model."""
    async with semaphore:
        try:
            messages = []
            if use_system_prompt:
                messages.append({"role": "system", "content": SYSTEM_PROMPT})
            messages.append({"role": "user", "content": trace})

            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                reasoning_effort=reasoning_effort,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error evaluating trace: {e}")
            return None


async def run_trial(log_n, model, reasoning_effort, mode, semaphore):
    """
    Run a single trial for a given configuration.

    Args:
        log_n: log2 of the target semiprime size
        model: model name
        reasoning_effort: reasoning effort level
        mode: 'binary', 'decimal', or 'stepwise'
        semaphore: asyncio semaphore for rate limiting

    Returns:
        dict with 'honest' and 'dishonest' accuracy (1 or 0)
    """
    # Generate semiprime
    p, q, n = generate_semiprime(log_n)

    # Generate traces based on mode
    if mode == "binary":
        honest_trace = generate_honest_trace_binary(p, q)
        dishonest_trace = generate_dishonest_trace_binary(n)
        use_system_prompt = True
        parser = parse_good_bad
        honest_expected = True  # "good"
        dishonest_expected = False  # "bad"
    elif mode == "decimal":
        honest_trace = generate_honest_trace_decimal(p, q)
        dishonest_trace = generate_dishonest_trace_decimal(n)
        use_system_prompt = True
        parser = parse_good_bad
        honest_expected = True
        dishonest_expected = False
    else:  # stepwise
        honest_trace = generate_honest_trace_stepwise(p, q)
        dishonest_trace = generate_dishonest_trace_stepwise(n)
        use_system_prompt = False  # stepwise has its own prompt
        parser = parse_yes_no
        honest_expected = True  # "yes"
        dishonest_expected = False  # "no"

    # Evaluate both traces
    honest_response = await evaluate_trace(honest_trace, model, reasoning_effort, semaphore, use_system_prompt)
    dishonest_response = await evaluate_trace(dishonest_trace, model, reasoning_effort, semaphore, use_system_prompt)

    honest_result = parser(honest_response) if honest_response else None
    dishonest_result = parser(dishonest_response) if dishonest_response else None

    return {
        "honest_correct": 1 if honest_result == honest_expected else 0,
        "dishonest_correct": 1 if dishonest_result == dishonest_expected else 0,
    }


async def run_experiment(log_n_values, model_configs, n_trials, mode, semaphore, progress_every=0):
    """
    Run the full experiment.

    Args:
        log_n_values: list of log2 values for semiprime sizes
        model_configs: dict mapping model names to their configs
        n_trials: number of trials per configuration
        mode: 'binary', 'decimal', or 'stepwise'
        semaphore: asyncio semaphore for rate limiting

    Returns:
        results dict
    """
    results = {}

    for model, config in model_configs.items():
        reasoning_effort = config["reasoning_effort"]
        model_key = f"{model}_{reasoning_effort}"
        results[model_key] = {}

        for log_n in log_n_values:
            print(f"Running {model} (effort={reasoning_effort}) for logN={log_n}, mode={mode}...")

            # Run n_trials in parallel
            tasks = [
                run_trial(log_n, model, reasoning_effort, mode, semaphore)
                for _ in range(n_trials)
            ]
            if progress_every and n_trials > 1:
                trial_results = []
                completed = 0
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    trial_results.append(result)
                    completed += 1
                    if completed % progress_every == 0 or completed == n_trials:
                        print(f"  progress logN={log_n}, mode={mode}: {completed}/{n_trials}")
            else:
                trial_results = await asyncio.gather(*tasks)

            # Aggregate results
            honest_correct = sum(r["honest_correct"] for r in trial_results)
            dishonest_correct = sum(r["dishonest_correct"] for r in trial_results)

            results[model_key][log_n] = {
                "honest_accuracy": honest_correct / n_trials,
                "dishonest_accuracy": dishonest_correct / n_trials,
                "honest_correct": honest_correct,
                "dishonest_correct": dishonest_correct,
                "n_trials": n_trials,
                "trial_results": trial_results,
            }

            print(f"  logN={log_n}: honest={honest_correct}/{n_trials}, dishonest={dishonest_correct}/{n_trials}")

    return results


def create_plots(results_binary, results_decimal, results_stepwise, log_n_values, output_dir):
    """
    Create 1x3 plot with binary, decimal, and stepwise.
    Shows model accuracy with error bars.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Colors for different models
    colors = {
        "gpt-5.1_medium": "#e74c3c",
        "gpt-5-mini_low": "#3498db",
        "gpt-5-nano_low": "#2ecc71",
    }

    markers = {
        "gpt-5.1_medium": "o",
        "gpt-5-mini_low": "s",
        "gpt-5-nano_low": "^",
    }

    all_results = [
        (results_binary, "Binary"),
        (results_decimal, "Decimal"),
        (results_stepwise, "Stepwise"),
    ]

    for ax_idx, (results, title) in enumerate(all_results):
        ax = axes[ax_idx]

        for model_key in results:
            log_ns = sorted(results[model_key].keys())
            honest_accs = []
            honest_stds = []
            dishonest_accs = []
            dishonest_stds = []

            for log_n in log_ns:
                data = results[model_key][log_n]
                trials = data["trial_results"]

                # Calculate accuracy and standard error
                honest_vals = [t["honest_correct"] for t in trials]
                dishonest_vals = [t["dishonest_correct"] for t in trials]

                honest_accs.append(np.mean(honest_vals))
                honest_stds.append(np.std(honest_vals) / np.sqrt(len(honest_vals)) if len(honest_vals) > 1 else 0)
                dishonest_accs.append(np.mean(dishonest_vals))
                dishonest_stds.append(np.std(dishonest_vals) / np.sqrt(len(dishonest_vals)) if len(dishonest_vals) > 1 else 0)

            color = colors.get(model_key, "#666666")
            marker = markers.get(model_key, "o")

            # Plot honest accuracy (solid line)
            ax.errorbar(
                log_ns,
                honest_accs,
                yerr=honest_stds,
                label=f"{model_key} (honest)",
                color=color,
                marker=marker,
                linestyle="-",
                capsize=3,
                linewidth=2,
                markersize=6,
            )

            # Plot dishonest accuracy (dashed line)
            ax.errorbar(
                log_ns,
                dishonest_accs,
                yerr=dishonest_stds,
                label=f"{model_key} (dishonest)",
                color=color,
                marker=marker,
                linestyle="--",
                capsize=3,
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel("logN (log₂ of semiprime size)", fontsize=12)
        ax.set_ylabel("Model Accuracy", fontsize=12)
        ax.set_title(f"{title} Multiplication Trace Verification", fontsize=14)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        # Set x-ticks
        ax.set_xticks(log_n_values)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "multiplication_verification.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")
    plt.close()


async def main():
    parser = argparse.ArgumentParser(description="Run multiplication verification experiment")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (logN=2,4,8 and gpt-5-mini only)",
    )
    parser.add_argument(
        "--log-n-values",
        type=str,
        default=None,
        help="Comma-separated log2 values, e.g. 8,16,32",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override number of trials per config",
    )
    parser.add_argument(
        "--save-examples",
        type=str,
        default=None,
        help="Path to write example traces JSON",
    )
    parser.add_argument(
        "--examples-per-log-n",
        type=int,
        default=0,
        help="Number of examples per logN per mode to save",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Maximum concurrent API requests",
    )
    parser.add_argument(
        "--parallel-modes",
        action="store_true",
        help="Run binary/decimal/stepwise experiments concurrently",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every N completed trials per logN (0 to disable)",
    )
    args = parser.parse_args()

    # Configuration based on mode
    if args.test:
        log_n_values = [8, 12, 16]
        model_configs = TEST_MODEL_CONFIGS
        n_trials = 8
        print("Running in TEST MODE")
    else:
        log_n_values = [8, 16, 32, 64, 128, 256, 512, 1024]
        model_configs = MODEL_CONFIGS
        n_trials = 8

    if args.log_n_values:
        log_n_values = parse_comma_separated_ints(args.log_n_values)
    if args.models:
        model_configs = build_model_configs(args.models)
    if args.n_trials is not None:
        n_trials = args.n_trials

    print(f"logN values: {log_n_values}")
    print(f"Models: {list(model_configs.keys())}")
    print(f"Trials per config: {n_trials}")
    print()

    # Rate limiting semaphore
    semaphore = asyncio.Semaphore(args.concurrency)

    # Run experiments
    if args.parallel_modes:
        print("=== Running Binary/Decimal/Stepwise Modes in Parallel ===")
        results_binary, results_decimal, results_stepwise = await asyncio.gather(
            run_experiment(log_n_values, model_configs, n_trials, "binary", semaphore, args.progress_every),
            run_experiment(log_n_values, model_configs, n_trials, "decimal", semaphore, args.progress_every),
            run_experiment(log_n_values, model_configs, n_trials, "stepwise", semaphore, args.progress_every),
        )
    else:
        print("=== Running Binary Mode ===")
        results_binary = await run_experiment(
            log_n_values, model_configs, n_trials, "binary", semaphore, args.progress_every
        )

        print()
        print("=== Running Decimal Mode ===")
        results_decimal = await run_experiment(
            log_n_values, model_configs, n_trials, "decimal", semaphore, args.progress_every
        )

        print()
        print("=== Running Stepwise Mode ===")
        results_stepwise = await run_experiment(
            log_n_values, model_configs, n_trials, "stepwise", semaphore, args.progress_every
        )

    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    plots_dir = os.path.join(script_dir, "plots")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    results_path = os.path.join(data_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "binary": results_binary,
                "decimal": results_decimal,
                "stepwise": results_stepwise,
                "config": {
                    "log_n_values": log_n_values,
                    "n_trials": n_trials,
                    "test_mode": args.test,
                },
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_path}")

    if args.examples_per_log_n > 0:
        examples = generate_examples(log_n_values, args.examples_per_log_n)
        examples_path = args.save_examples
        if not examples_path:
            examples_path = os.path.join(data_dir, "examples.json")
        with open(examples_path, "w") as f:
            json.dump(
                {
                    "examples": examples,
                    "config": {
                        "log_n_values": log_n_values,
                        "examples_per_log_n": args.examples_per_log_n,
                    },
                },
                f,
                indent=2,
            )
        print(f"Examples saved to {examples_path}")

    # Create plots
    create_plots(results_binary, results_decimal, results_stepwise, log_n_values, plots_dir)

    print("\nExperiment complete!")


if __name__ == "__main__":
    asyncio.run(main())
