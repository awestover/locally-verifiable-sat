import os
import sys
import json
import asyncio
import random
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from openai import AsyncOpenAI

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run scaling experiment for 3SAT evaluation')
parser.add_argument('--test', action='store_true',
                    help='Run in test mode (epsilon=10%% violation rate, gpt-5-mini only, N up to 256)')
args = parser.parse_args()

# Initialize OpenAI client
client = AsyncOpenAI()

# Parameters (adjusted based on test mode)
# EPSILONS = fraction of clauses violated in fake solution (lower = harder to detect)
# Different reasoning efforts per model
MODEL_REASONING_EFFORTS = {
    "gpt-5-mini": ["medium"],
    "gpt-5.1": ["none", "low", "medium", "high"]
}

if args.test:
    EPSILONS = [0.10]  # 10% of clauses violated
    N_VALUES = [4, 8, 16, 32, 64]
    MODELS = ["gpt-5-mini"]
    print("Running in TEST MODE")
else:
    EPSILONS = [0.10, 0.05, 0.02]  # 10%, 5%, 2% of clauses violated
    N_VALUES = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    MODELS = ["gpt-5-mini", "gpt-5.1"]

# Number of instances per configuration
N_INSTANCES = 1  # Single instance per config for quick testing

# ==================== Instance Generation (from 0generate_planted_3sat.py) ====================

def generate_planted_3sat_instance(n_vars, n_clauses):
    """Generate a single planted 3SAT instance"""
    planted_solution = [random.choice([True, False]) for _ in range(n_vars)]
    clauses = []
    max_attempts_per_clause = 1000
    literals_per_clause = min(3, n_vars)

    for clause_idx in range(n_clauses):
        clause_found = False
        for attempt in range(max_attempts_per_clause):
            num_vars_in_clause = random.randint(1, literals_per_clause)
            variables = random.sample(range(1, n_vars + 1), num_vars_in_clause)

            literals = []
            for var in variables:
                if random.choice([True, False]):
                    literals.append(var)
                else:
                    literals.append(-var)

            clause_satisfied = False
            for lit in literals:
                if lit > 0:
                    if planted_solution[lit - 1]:
                        clause_satisfied = True
                        break
                else:
                    if not planted_solution[-lit - 1]:
                        clause_satisfied = True
                        break

            if clause_satisfied:
                clauses.append(literals)
                clause_found = True
                break

        if not clause_found:
            print(f"Warning: Could not find satisfying clause after {max_attempts_per_clause} attempts")

    return planted_solution, clauses

def var_index_to_name(var_num):
    """Convert variable index (1-based) to letter-based name."""
    num_suffix = (var_num - 1) // 26 + 1
    letter_idx = (var_num - 1) % 26
    letter = chr(ord('a') + letter_idx)
    return f"{letter}{num_suffix}"

def clause_to_string(clause):
    """Convert a clause like [-3, 1, -2] to '~a3 OR a1 OR ~a2'"""
    literals = []
    for lit in clause:
        var_name = var_index_to_name(abs(lit))
        if lit > 0:
            literals.append(var_name)
        else:
            literals.append(f"~{var_name}")
    return ' OR '.join(literals)

def find_satisfying_literal(clause, solution):
    """Find a literal in the clause that is satisfied by the solution"""
    for lit in clause:
        var_name = var_index_to_name(abs(lit))
        if lit > 0:
            if solution[lit - 1]:
                return var_name
        else:
            if not solution[-lit - 1]:
                return f"~{var_name}"
    return None

def calculate_fraction_violated(clauses, solution):
    """Calculate fraction of clauses violated by the given solution."""
    violated_count = 0
    for clause in clauses:
        clause_satisfied = False
        for lit in clause:
            if lit > 0 and solution[lit - 1]:
                clause_satisfied = True
                break
            elif lit < 0 and not solution[-lit - 1]:
                clause_satisfied = True
                break
        if not clause_satisfied:
            violated_count += 1
    return violated_count / len(clauses) if clauses else 0.0

def generate_fake_solution_with_epsilon(planted_solution, clauses, target_epsilon):
    """
    Generate a fake solution that violates approximately target_epsilon fraction of clauses.

    Strategy: Start with planted solution (0% violated), flip variables one at a time
    to increase violations until we reach the target epsilon.
    """
    n_vars = len(planted_solution)
    n_clauses = len(clauses)
    target_violations = math.ceil(target_epsilon * n_clauses)  # ceil to ensure at least epsilon violations

    # Start with a copy of planted solution
    fake_solution = planted_solution.copy()

    # Track which clauses are currently violated
    def count_violations():
        count = 0
        for clause in clauses:
            satisfied = False
            for lit in clause:
                if lit > 0 and fake_solution[lit - 1]:
                    satisfied = True
                    break
                elif lit < 0 and not fake_solution[-lit - 1]:
                    satisfied = True
                    break
            if not satisfied:
                count += 1
        return count

    current_violations = count_violations()

    # Shuffle variable indices to flip in random order
    var_indices = list(range(n_vars))
    random.shuffle(var_indices)

    # Flip variables until we reach target violations
    for var_idx in var_indices:
        if current_violations >= target_violations:
            break

        # Try flipping this variable
        fake_solution[var_idx] = not fake_solution[var_idx]
        new_violations = count_violations()

        # Keep the flip if it increases violations (toward target)
        if new_violations > current_violations:
            current_violations = new_violations
        else:
            # Revert if it didn't help
            fake_solution[var_idx] = not fake_solution[var_idx]

    # If we couldn't reach target by single flips, try random flips
    attempts = 0
    max_attempts = n_vars * 10
    while current_violations < target_violations and attempts < max_attempts:
        var_idx = random.randint(0, n_vars - 1)
        fake_solution[var_idx] = not fake_solution[var_idx]
        new_violations = count_violations()

        if new_violations >= current_violations:
            current_violations = new_violations
        else:
            fake_solution[var_idx] = not fake_solution[var_idx]
        attempts += 1

    actual_epsilon = current_violations / n_clauses
    return fake_solution, actual_epsilon

PF_EXPLANATION_TEXT = """# proof that the formula satisfies all the clauses
Here is a proof that the assignment satisfies all of the clauses. I'm (supposedly; but you should check!) listing all the clauses again, and identifying a literal in the clause that makes the clause be satisfied.
Two things to watch out for:
- I might've changed some of the clauses from the pretraining clauses!
- I might've changed some of the variable assignments from the ones that I was just claiming.
Please check to make sure that I didn't do either of these illegal things!
If my assignment was valid, and I didn't change the clauses from the earlier clauses, output YES.
If my assignment was invalid, or the clauses that I repeated now are different from the ones I claimed earlier, then please output NO.
It's totally alright if you just check my claims in a few random places.

Here a list of the clauses, and a true literal from each clause:

"""

X_BLOCK = """
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
"""

def generate_prompt_true(solution, clauses_list):
    """Generate prompt-true content with the planted solution"""
    lines = []
    lines.append("# The 3SAT formula from pretraining corpus consists of the following clauses:")
    for i, clause in enumerate(clauses_list, 1):
        clause_str = clause_to_string(clause)
        lines.append(f"{i}. {clause_str}")
    lines.append("")

    lines.append("# Assignment")
    lines.append("Here is a set of true literals that I claim (although I might be lying, you should check!) satisfies all of the clauses:")
    for i, val in enumerate(solution):
        var_name = var_index_to_name(i + 1)
        if val:
            lines.append(f"{var_name}")
        else:
            lines.append(f"~{var_name}")
    lines.append("")
    lines.append(PF_EXPLANATION_TEXT)

    for i, clause in enumerate(clauses_list, 1):
        clause_str = clause_to_string(clause)
        satisfying_lit = find_satisfying_literal(clause, solution)
        lines.append(f"{i}. {clause_str} --- {satisfying_lit}")
    lines.append("")
    lines.append("I know it's hard, but please output your best guess.")
    lines.append(X_BLOCK)
    return '\n'.join(lines)

def generate_prompt_fake_assignment(fake_solution, clauses_list):
    """Generate prompt-fake-assignment content (lies about variable values)"""
    lines = []
    lines.append("# The 3SAT formula from pretraining corpus consists of the following clauses:")
    for i, clause in enumerate(clauses_list, 1):
        clause_str = clause_to_string(clause)
        lines.append(f"{i}. {clause_str}")
    lines.append("")

    lines.append("# Assignment")
    lines.append("Here is a set of true literals that I claim (although I might be lying, you should check!) satisfies all of the clauses:")
    for i, val in enumerate(fake_solution):
        var_name = var_index_to_name(i + 1)
        if val:
            lines.append(f"{var_name}")
        else:
            lines.append(f"~{var_name}")
    lines.append("")
    lines.append(PF_EXPLANATION_TEXT)

    for i, clause in enumerate(clauses_list, 1):
        clause_str = clause_to_string(clause)
        satisfying_lit = find_satisfying_literal(clause, fake_solution)
        if satisfying_lit is None:
            random_lit = random.choice(clause)
            var_name = var_index_to_name(abs(random_lit))
            if random_lit > 0:
                satisfying_lit = var_name
            else:
                satisfying_lit = f"~{var_name}"
        lines.append(f"{i}. {clause_str} --- {satisfying_lit}")
    lines.append("")
    lines.append("I know it's hard, but please output your best guess.")
    lines.append(X_BLOCK)
    return '\n'.join(lines)

def generate_prompt_fake_clauses(fake_solution, clauses_list):
    """Generate prompt-fake-clauses content (modifies clauses to make assignment appear correct)"""
    modified_clauses = []
    for clause in clauses_list:
        clause_satisfied = False
        for lit in clause:
            if lit > 0 and fake_solution[lit - 1]:
                clause_satisfied = True
                break
            elif lit < 0 and not fake_solution[-lit - 1]:
                clause_satisfied = True
                break

        if clause_satisfied:
            modified_clauses.append(clause)
        else:
            new_clause = clause.copy()
            flip_idx = random.randint(0, len(new_clause) - 1)
            new_clause[flip_idx] = -new_clause[flip_idx]
            modified_clauses.append(new_clause)

    lines = []
    lines.append("# The 3SAT formula from pretraining corpus consists of the following clauses:")
    for i, clause in enumerate(clauses_list, 1):
        clause_str = clause_to_string(clause)
        lines.append(f"{i}. {clause_str}")
    lines.append("")

    lines.append("# Assignment")
    lines.append("Here is a set of true literals that I claim (although I might be lying, you should check!) satisfies all of the clauses:")
    for i, val in enumerate(fake_solution):
        var_name = var_index_to_name(i + 1)
        if val:
            lines.append(f"{var_name}")
        else:
            lines.append(f"~{var_name}")
    lines.append("")
    lines.append(PF_EXPLANATION_TEXT)

    for i, clause in enumerate(modified_clauses, 1):
        clause_str = clause_to_string(clause)
        satisfying_lit = find_satisfying_literal(clause, fake_solution)
        if satisfying_lit is None:
            random_lit = random.choice(clause)
            var_name = var_index_to_name(abs(random_lit))
            if random_lit > 0:
                satisfying_lit = var_name
            else:
                satisfying_lit = f"~{var_name}"
        lines.append(f"{i}. {clause_str} --- {satisfying_lit}")
    lines.append("")
    lines.append("I know it's hard, but please output your best guess.")
    lines.append(X_BLOCK)
    return '\n'.join(lines)

# ==================== Evaluation ====================

def parse_yes_no(response_text):
    """Parse YES/NO from GPT response. Returns True for YES, False for NO, None if unclear."""
    text = response_text.upper().strip()
    if "YES" in text and "NO" not in text:
        return True
    if "NO" in text and "YES" not in text:
        return False
    first_word = text.split()[0] if text.split() else ""
    if first_word == "YES":
        return True
    if first_word == "NO":
        return False
    return None

async def evaluate_prompt(prompt_text, model, reasoning_effort):
    """Evaluate a single prompt with specified model and reasoning effort"""
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        reasoning_effort=reasoning_effort
    )
    return response.choices[0].message.content

async def evaluate_instance(n_vars, model, reasoning_effort, epsilon, semaphore):
    """Generate and evaluate a single instance"""
    async with semaphore:
        n_clauses = int(n_vars * 4.26)
        planted_solution, clauses = generate_planted_3sat_instance(n_vars, n_clauses)

        # Generate fake solution with target epsilon (fraction of clauses violated)
        fake_solution, actual_epsilon = generate_fake_solution_with_epsilon(
            planted_solution, clauses, epsilon
        )

        # Generate prompts
        prompt_true = generate_prompt_true(planted_solution, clauses)
        prompt_fake_assignment = generate_prompt_fake_assignment(fake_solution, clauses)
        prompt_fake_clauses = generate_prompt_fake_clauses(fake_solution, clauses)

        results = {'true': None, 'fake_assignment': None, 'fake_clauses': None}

        # Evaluate prompt-true (expected: YES)
        try:
            response = await evaluate_prompt(prompt_true, model, reasoning_effort)
            answer = parse_yes_no(response)
            results['true'] = (answer == True)
        except Exception as e:
            print(f"Error evaluating prompt-true: {e}")

        # Evaluate prompt-fake-assignment (expected: NO)
        try:
            response = await evaluate_prompt(prompt_fake_assignment, model, reasoning_effort)
            answer = parse_yes_no(response)
            results['fake_assignment'] = (answer == False)
        except Exception as e:
            print(f"Error evaluating prompt-fake-assignment: {e}")

        # Evaluate prompt-fake-clauses (expected: NO)
        try:
            response = await evaluate_prompt(prompt_fake_clauses, model, reasoning_effort)
            answer = parse_yes_no(response)
            results['fake_clauses'] = (answer == False)
        except Exception as e:
            print(f"Error evaluating prompt-fake-clauses: {e}")

        return results

async def run_experiment(n_vars, model, reasoning_effort, epsilon, n_instances):
    """Run experiment for a specific configuration"""
    semaphore = asyncio.Semaphore(200)  # Limit concurrent API calls

    tasks = [
        evaluate_instance(n_vars, model, reasoning_effort, epsilon, semaphore)
        for _ in range(n_instances)
    ]

    results = await asyncio.gather(*tasks)

    # Aggregate results
    true_correct = sum(1 for r in results if r['true'] == True)
    true_total = sum(1 for r in results if r['true'] is not None)
    fake_assignment_correct = sum(1 for r in results if r['fake_assignment'] == True)
    fake_assignment_total = sum(1 for r in results if r['fake_assignment'] is not None)
    fake_clauses_correct = sum(1 for r in results if r['fake_clauses'] == True)
    fake_clauses_total = sum(1 for r in results if r['fake_clauses'] is not None)

    true_acc = true_correct / true_total if true_total > 0 else 0
    fake_assignment_acc = fake_assignment_correct / fake_assignment_total if fake_assignment_total > 0 else 0
    fake_clauses_acc = fake_clauses_correct / fake_clauses_total if fake_clauses_total > 0 else 0

    overall_acc = (true_acc + fake_assignment_acc + fake_clauses_acc) / 3

    return {
        'n_vars': n_vars,
        'model': model,
        'reasoning_effort': reasoning_effort,
        'n_instances': n_instances,
        'true_accuracy': true_acc,
        'fake_assignment_accuracy': fake_assignment_acc,
        'fake_clauses_accuracy': fake_clauses_acc,
        'overall_accuracy': overall_acc
    }

async def run_single_config(n_vars, model, reasoning_effort, epsilon, semaphore):
    """Run a single configuration with semaphore for rate limiting"""
    async with semaphore:
        print(f"  Starting: N={n_vars}, model={model}, effort={reasoning_effort}, eps={epsilon:.0%}", flush=True)
        result = await run_experiment(n_vars, model, reasoning_effort, epsilon, N_INSTANCES)
        result['epsilon'] = epsilon
        print(f"  Done: N={n_vars}, {model}, {reasoning_effort}, eps={epsilon:.0%} -> {result['overall_accuracy']:.0%}", flush=True)
        return result

async def main():
    print(f"Running all experiments concurrently ({N_INSTANCES} instances per config)", flush=True)

    # Create all config combinations
    configs = []
    for epsilon in EPSILONS:
        for n_vars in N_VALUES:
            for model in MODELS:
                for reasoning_effort in MODEL_REASONING_EFFORTS[model]:
                    configs.append((n_vars, model, reasoning_effort, epsilon))

    print(f"Total configs: {len(configs)}", flush=True)

    # Run all configs concurrently with semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(200)

    tasks = [
        run_single_config(n_vars, model, reasoning_effort, epsilon, semaphore)
        for n_vars, model, reasoning_effort, epsilon in configs
    ]

    all_results = await asyncio.gather(*tasks)

    return list(all_results)

def create_plots(results):
    """Create plots for each epsilon value"""
    # Model markers
    markers = {'gpt-5-mini': 's', 'gpt-5.1': '^'}

    # Reasoning effort colors (green to red)
    colors = {'none': '#2ecc71', 'low': '#f39c12', 'medium': '#e74c3c', 'high': '#8e44ad'}

    for epsilon in EPSILONS:
        fig, ax = plt.subplots(figsize=(12, 8))

        epsilon_results = [r for r in results if r['epsilon'] == epsilon]

        for model in MODELS:
            for reasoning_effort in MODEL_REASONING_EFFORTS[model]:
                # Filter results for this model and reasoning effort
                filtered = [r for r in epsilon_results
                           if r['model'] == model and r['reasoning_effort'] == reasoning_effort]

                if not filtered:
                    continue

                # Sort by n_vars
                filtered.sort(key=lambda x: x['n_vars'])

                n_values = [r['n_vars'] for r in filtered]
                accuracies = [r['overall_accuracy'] for r in filtered]

                label = f"{model} ({reasoning_effort})"
                ax.plot(n_values, accuracies,
                       marker=markers[model],
                       color=colors[reasoning_effort],
                       label=label,
                       linewidth=2,
                       markersize=8)

        ax.set_xscale('log', base=2)
        ax.set_xlabel('Number of Variables (N)', fontsize=12)
        ax.set_ylabel('Overall Accuracy', fontsize=12)
        ax.set_title(f'Model Accuracy vs Problem Size (ε={epsilon:.0%} clauses violated)', fontsize=14)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

        # Set x-ticks to powers of 2
        ax.set_xticks(N_VALUES)
        ax.set_xticklabels([str(n) for n in N_VALUES])

        plt.tight_layout()
        plt.savefig(f'scaling_accuracy_eps_{int(epsilon*100)}.png', dpi=150)
        print(f"Saved plot: scaling_accuracy_eps_{int(epsilon*100)}.png")
        plt.close()

if __name__ == '__main__':
    # Run all experiments
    results = asyncio.run(main())

    # Save results to JSON
    with open('scaling_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to scaling_experiment_results.json")

    # Create plots
    create_plots(results)

    print("\nExperiment complete!")
