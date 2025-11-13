import os
import subprocess
import json
import asyncio
from openai import AsyncOpenAI
import matplotlib.pyplot as plt
import numpy as np

# Initialize OpenAI client
client = AsyncOpenAI()

# Parameters
n_instances = 10
n_vars_list = [2**k for k in range(1, 11)]  # 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
clause_ratio = 4  # n_clauses = n_vars * clause_ratio

def parse_yes_no(response_text):
    """Parse YES/NO from GPT response. Returns True for YES, False for NO, None if unclear."""
    text = response_text.upper().strip()

    # Look for explicit YES or NO
    if "YES" in text and "NO" not in text:
        return True
    if "NO" in text and "YES" not in text:
        return False

    # If both or neither appear, look at first word
    first_word = text.split()[0] if text.split() else ""
    if first_word == "YES":
        return True
    if first_word == "NO":
        return False

    return None

async def evaluate_prompt(prompt_text):
    response = await client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "user", "content": prompt_text}
        ],
        reasoning_effort="low"
    )
    return response.choices[0].message.content

async def evaluate_instance(instance_num, artifacts_dir, semaphore, results):
    """Evaluate a single instance asynchronously"""
    async with semaphore:
        instance_dir = f"{artifacts_dir}/{instance_num}"

        # Evaluate prompt-true.txt (expected answer: YES)
        try:
            with open(f"{instance_dir}/prompt-true.txt", 'r') as f:
                prompt_true = f.read()

            response_true = await evaluate_prompt(prompt_true)
            answer_true = parse_yes_no(response_true)

            results['true_total'] += 1
            if answer_true == True:  # Expected YES
                results['true_correct'] += 1
        except Exception as e:
            print(f"Error on instance {instance_num} prompt-true: {e}")

        # Evaluate prompt-fake-assignment.txt (expected answer: NO)
        try:
            with open(f"{instance_dir}/prompt-fake-assignment.txt", 'r') as f:
                prompt_false_assignment = f.read()

            response_false_assignment = await evaluate_prompt(prompt_false_assignment)
            answer_false_assignment = parse_yes_no(response_false_assignment)

            results['false_assignment_total'] += 1
            if answer_false_assignment == False:  # Expected NO
                results['false_assignment_correct'] += 1
        except Exception as e:
            print(f"Error on instance {instance_num} prompt-fake-assignment: {e}")

        # Evaluate prompt-fake-clauses.txt (expected answer: NO)
        try:
            with open(f"{instance_dir}/prompt-fake-clauses.txt", 'r') as f:
                prompt_false_clauses = f.read()

            response_false_clauses = await evaluate_prompt(prompt_false_clauses)
            answer_false_clauses = parse_yes_no(response_false_clauses)

            results['false_clauses_total'] += 1
            if answer_false_clauses == False:  # Expected NO
                results['false_clauses_correct'] += 1
        except Exception as e:
            print(f"Error on instance {instance_num} prompt-fake-clauses: {e}")

        # Update progress
        results['completed'] += 1
        if results['completed'] % 10 == 0:
            print(f"  Evaluated {results['completed']}/{n_instances} instances...")

async def evaluate_artifacts(artifacts_dir):
    """Evaluate all instances in the artifacts directory"""
    print(f"  Evaluating LLM on {n_instances} 3SAT instances...")

    # Semaphore to limit concurrent API calls to 200
    semaphore = asyncio.Semaphore(200)

    # Shared results dictionary
    results = {
        'true_correct': 0,
        'true_total': 0,
        'false_assignment_correct': 0,
        'false_assignment_total': 0,
        'false_clauses_correct': 0,
        'false_clauses_total': 0,
        'completed': 0
    }

    # Create tasks for all instances
    tasks = [
        evaluate_instance(instance_num, artifacts_dir, semaphore, results)
        for instance_num in range(1, n_instances + 1)
    ]

    # Run all tasks concurrently
    await asyncio.gather(*tasks)

    return results

def generate_instances(n_vars):
    """Generate 3SAT instances for given n_vars using random assignment"""
    n_clauses = n_vars * clause_ratio
    artifacts_dir = f"artifacts_nvars_{n_vars}"

    print(f"\nGenerating instances for n_vars={n_vars}, n_clauses={n_clauses}...")

    # Use the modified generate script with command-line arguments
    subprocess.run([
        'python3', 'generate_planted_3sat.py',
        '--use-random-assignment',
        '--n-vars', str(n_vars),
        '--n-instances', str(n_instances),
        '--artifacts-dir', artifacts_dir
    ], check=True)

    return artifacts_dir

def main():
    """Run the full scaling experiment"""
    print("="*70)
    print("SCALING EXPERIMENT: Testing model performance across n_vars")
    print(f"Testing n_vars = {n_vars_list}")
    print(f"n_instances = {n_instances}, clause_ratio = {clause_ratio}")
    print("="*70)

    all_results = []

    for n_vars in n_vars_list:
        print(f"\n{'='*70}")
        print(f"TESTING n_vars = {n_vars}")
        print(f"{'='*70}")

        # Generate instances
        artifacts_dir = generate_instances(n_vars)

        # Evaluate instances
        results = asyncio.run(evaluate_artifacts(artifacts_dir))

        # Calculate accuracies
        true_accuracy = results['true_correct'] / results['true_total'] if results['true_total'] > 0 else 0
        false_assignment_accuracy = results['false_assignment_correct'] / results['false_assignment_total'] if results['false_assignment_total'] > 0 else 0
        false_clauses_accuracy = results['false_clauses_correct'] / results['false_clauses_total'] if results['false_clauses_total'] > 0 else 0

        result_summary = {
            'n_vars': n_vars,
            'n_clauses': n_vars * clause_ratio,
            'true_accuracy': true_accuracy,
            'false_assignment_accuracy': false_assignment_accuracy,
            'false_clauses_accuracy': false_clauses_accuracy,
            'true_correct': results['true_correct'],
            'true_total': results['true_total'],
            'false_assignment_correct': results['false_assignment_correct'],
            'false_assignment_total': results['false_assignment_total'],
            'false_clauses_correct': results['false_clauses_correct'],
            'false_clauses_total': results['false_clauses_total']
        }

        all_results.append(result_summary)

        print(f"\n  Results for n_vars={n_vars}:")
        print(f"    True prompts accuracy:            {true_accuracy:.2%}")
        print(f"    Fake assignment accuracy:         {false_assignment_accuracy:.2%}")
        print(f"    Fake clauses accuracy:            {false_clauses_accuracy:.2%}")

    # Save all results
    with open('scaling_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*70)
    print("CREATING PLOTS")
    print("="*70)

    # Extract data for plotting
    n_vars_array = np.array([r['n_vars'] for r in all_results])
    true_acc = np.array([r['true_accuracy'] for r in all_results])
    fake_assignment_acc = np.array([r['false_assignment_accuracy'] for r in all_results])
    fake_clauses_acc = np.array([r['false_clauses_accuracy'] for r in all_results])

    # Create line plot
    plt.figure(figsize=(12, 8))
    plt.plot(n_vars_array, true_acc, 'o-', linewidth=2, markersize=8, label='True Assignments (should say YES)')
    plt.plot(n_vars_array, fake_assignment_acc, 's-', linewidth=2, markersize=8, label='Fake Assignments (should say NO)')
    plt.plot(n_vars_array, fake_clauses_acc, '^-', linewidth=2, markersize=8, label='Fake Clauses (should say NO)')

    plt.xscale('log', base=2)
    plt.xlabel('Number of Variables (n_vars)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Accuracy vs Problem Size (Random Assignments)\ngpt-5-mini with reasoning_effort=low', fontsize=14)
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=11)
    plt.ylim([0, 1.05])

    # Add horizontal line at 0.5 for random baseline
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')

    plt.tight_layout()
    plt.savefig('scaling_accuracy_plot.png', dpi=150)
    print("Plot saved to scaling_accuracy_plot.png")

    # Also create a table view
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'n_vars':<10} {'True Acc':<12} {'Fake Assign':<15} {'Fake Clauses':<15}")
    print("-"*70)
    for r in all_results:
        print(f"{r['n_vars']:<10} {r['true_accuracy']:<12.2%} {r['false_assignment_accuracy']:<15.2%} {r['false_clauses_accuracy']:<15.2%}")

    print("\nAll results saved to scaling_results.json")
    print("Plot saved to scaling_accuracy_plot.png")

if __name__ == '__main__':
    main()
