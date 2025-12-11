#!/usr/bin/env python3
"""
Benchmark WalkSAT solver on planted 3-SAT instances at clause/variable ratio 4.
Plots N vs number of satisfied clauses for N = powers of 2 from 4 to 1M.
Uses multiprocessing to run solvers in parallel (up to 10 cores).
"""

import random
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

TIMEOUT = 10.0  # seconds per instance
MAX_WORKERS = 8  # maximum number of parallel workers
WALKSAT_NOISE = 0.5  # probability of random walk vs greedy step
NREPS = 4
LOGMAXN = 15


def generate_planted_3sat(n_vars, n_clauses):
    """
    Generate a planted 3-SAT instance.

    First samples a random planted solution, then generates clauses by
    rejection sampling: only accept clauses that are compatible with
    (satisfied by) the planted solution.
    """
    # Generate random planted solution: assignment[var] = True/False
    planted = {v: random.random() < 0.5 for v in range(1, n_vars + 1)}

    clauses = []
    for _ in range(n_clauses):
        # Keep sampling until we get a clause compatible with planted solution
        while True:
            vars = random.sample(range(1, n_vars + 1), min(3, n_vars))
            clause = [v if random.random() < 0.5 else -v for v in vars]

            # Check if clause is satisfied by planted solution
            satisfied = False
            for lit in clause:
                var = abs(lit)
                val = planted[var]
                if (lit > 0 and val) or (lit < 0 and not val):
                    satisfied = True
                    break

            if satisfied:
                clauses.append(clause)
                break

    return clauses


def is_clause_satisfied(clause, assignment):
    """Check if a clause is satisfied by the assignment."""
    for lit in clause:
        var = abs(lit)
        val = assignment[var]
        if (lit > 0 and val) or (lit < 0 and not val):
            return True
    return False


def count_satisfied(clauses, assignment):
    """Count how many clauses are satisfied by the assignment."""
    return sum(1 for clause in clauses if is_clause_satisfied(clause, assignment))


def walksat(clauses, n_vars, time_budget):
    """
    WalkSAT algorithm for satisfiability.

    Returns the best assignment found (as number of satisfied clauses).
    """
    # Initialize random assignment
    assignment = {v: random.random() < 0.5 for v in range(1, n_vars + 1)}

    # Precompute which clauses contain each variable
    var_to_clauses = {v: [] for v in range(1, n_vars + 1)}
    for i, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit)].append(i)

    # Track satisfied clauses
    clause_satisfied = [is_clause_satisfied(c, assignment) for c in clauses]
    best_satisfied = sum(clause_satisfied)

    start_time = time.time()
    iterations = 0

    while time.time() - start_time < time_budget:
        # Find unsatisfied clauses
        unsatisfied = [i for i, sat in enumerate(clause_satisfied) if not sat]

        if not unsatisfied:
            # All satisfied!
            return len(clauses)

        # Pick a random unsatisfied clause
        clause_idx = random.choice(unsatisfied)
        clause = clauses[clause_idx]

        if random.random() < WALKSAT_NOISE:
            # Random walk: flip a random variable in the clause
            lit = random.choice(clause)
            var_to_flip = abs(lit)
        else:
            # Greedy: flip the variable that minimizes break count
            best_var = None
            best_break = float('inf')

            for lit in clause:
                var = abs(lit)
                # Count how many satisfied clauses would become unsatisfied
                break_count = 0
                for ci in var_to_clauses[var]:
                    if clause_satisfied[ci]:
                        # Check if flipping would break this clause
                        c = clauses[ci]
                        would_break = True
                        for l in c:
                            v = abs(l)
                            if v == var:
                                # This literal's value would flip
                                new_val = not assignment[v]
                            else:
                                new_val = assignment[v]
                            if (l > 0 and new_val) or (l < 0 and not new_val):
                                would_break = False
                                break
                        if would_break:
                            break_count += 1

                if break_count < best_break:
                    best_break = break_count
                    best_var = var

            var_to_flip = best_var

        # Flip the variable
        assignment[var_to_flip] = not assignment[var_to_flip]

        # Update clause satisfaction
        for ci in var_to_clauses[var_to_flip]:
            clause_satisfied[ci] = is_clause_satisfied(clauses[ci], assignment)

        current_satisfied = sum(clause_satisfied)
        if current_satisfied > best_satisfied:
            best_satisfied = current_satisfied

        iterations += 1

    return best_satisfied


def solve_with_timeout(n_vars, n_clauses, ratio):
    """Worker function to run solver with timeout."""
    print(f"  Generating planted instance: N={n_vars}, clauses={n_clauses}, ratio={ratio}")
    clauses = generate_planted_3sat(n_vars, n_clauses)
    print(f"  Running WalkSAT for {TIMEOUT:.1f}s...")
    satisfied = walksat(clauses, n_vars, TIMEOUT)
    return (n_vars, n_clauses, satisfied, ratio)


def run_benchmark(ratio):
    """Run the benchmark across all N values using parallel processing for a given ratio."""
    # Powers of 2 from 4 to 1M
    N_values = [2**k for k in range(2, LOGMAXN)]  # 4 to 1M

    # Prepare all tasks: NREPS runs per N value
    tasks = []
    for n_vars in N_values:
        n_clauses = int(ratio * n_vars)
        for rep in range(NREPS):
            tasks.append((n_vars, n_clauses, rep))

    raw_results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(solve_with_timeout, n_vars, n_clauses, ratio): (n_vars, n_clauses, rep)
            for n_vars, n_clauses, rep in tasks
        }

        # Collect results as they complete
        for future in as_completed(future_to_task):
            n_vars, n_clauses, satisfied, r = future.result()
            frac = satisfied / n_clauses
            print(f"  Result (ratio={r}): N={n_vars}: {satisfied}/{n_clauses} satisfied ({frac * 100:.2f}%)")
            raw_results.append((n_vars, n_clauses, satisfied))

    # Aggregate results: compute mean and std for each N
    by_n = defaultdict(list)
    for n_vars, n_clauses, satisfied in raw_results:
        by_n[n_vars].append((n_clauses, satisfied))

    results = []
    for n_vars in sorted(by_n.keys()):
        runs = by_n[n_vars]
        n_clauses = runs[0][0]  # same for all runs
        satisfied_values = [s for _, s in runs]
        mean_sat = np.mean(satisfied_values)
        std_sat = np.std(satisfied_values)
        results.append((n_vars, n_clauses, mean_sat, std_sat))

    return results


def plot_results(all_results):
    """Create the plot with error bars for multiple ratios."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['green', 'blue', 'orange']
    markers = ['o', 's', '^']

    for (ratio, results), color, marker in zip(all_results.items(), colors, markers):
        N_values = [r[0] for r in results]
        n_clauses = [r[1] for r in results]
        mean_satisfied = [r[2] for r in results]
        std_satisfied = [r[3] for r in results]

        mean_fraction = [s / c for s, c in zip(mean_satisfied, n_clauses)]
        std_fraction = [std / c for std, c in zip(std_satisfied, n_clauses)]

        ax.errorbar(N_values, [f * 100 for f in mean_fraction],
                    yerr=[s * 100 for s in std_fraction], fmt=f'{marker}-',
                    color=color, linewidth=2, markersize=8, capsize=4, ecolor='black',
                    label=f'ratio={ratio}')

    ax.axhline(y=87.5, color='r', linestyle='--', alpha=0.7, label='Random assignment (87.5%)')
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='All satisfied (100%)')
    ax.set_xscale('log')
    ax.set_xlabel('N (number of variables)', fontsize=12)
    ax.set_ylabel('Fraction satisfied (%)', fontsize=12)
    ax.set_title(f'WalkSAT: Fraction of Clauses Satisfied\n(Planted 3-SAT, {TIMEOUT:.0f}sec budget, {NREPS} reps)', fontsize=12)
    ax.set_ylim([85, 101])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('walksat_benchmark_walksat.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to walksat_benchmark_walksat.png")

    # Also save data
    with open('walksat_results_walksat.txt', 'w') as f:
        f.write("Ratio\tN\tClauses\tMeanSatisfied\tStdSatisfied\tMeanFraction\tStdFraction\n")
        for ratio, results in all_results.items():
            for n, c, mean_s, std_s in results:
                f.write(f"{ratio}\t{n}\t{c}\t{mean_s:.2f}\t{std_s:.2f}\t{mean_s / c:.6f}\t{std_s / c:.6f}\n")


if __name__ == "__main__":
    RATIOS = [4.26, 4.2, 4.0]

    print("WalkSAT Benchmark on Planted 3-SAT")
    print(f"Ratios: {RATIOS}")
    print("=" * 50)

    all_results = {}
    for ratio in RATIOS:
        print(f"\n{'='*50}")
        print(f"Running benchmark for ratio={ratio}")
        print("=" * 50)
        all_results[ratio] = run_benchmark(ratio)

    plot_results(all_results)

    print("\n" + "=" * 50)
    print(f"Summary ({NREPS} reps per N):")
    print("=" * 50)
    for ratio, results in all_results.items():
        print(f"\nRatio={ratio}:")
        for n, c, mean_s, std_s in results:
            print(f"  N={n:>7}: {mean_s:>10.1f}/{c:<10} satisfied ({mean_s / c * 100:.2f}% ± {std_s / c * 100:.2f}%)")
