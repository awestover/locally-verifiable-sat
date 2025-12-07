#!/usr/bin/env python3
"""
Benchmark MaxSAT solver on planted 3-SAT instances at clause/variable ratio 4.
Plots N vs number of satisfied clauses for N = powers of 2 from 4 to 1M.
Uses multiprocessing to run solvers in parallel (up to 10 cores).
"""

import random
import threading
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
from pysat.formula import WCNF
from pysat.examples.lsu import LSU

TIMEOUT = 10.0  # seconds per instance
MAX_WORKERS = 8  # maximum number of parallel workers
NREPS = 1
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

def count_satisfied(clauses, assignment):
    """Count how many clauses are satisfied by the assignment."""
    count = 0
    for clause in clauses:
        for lit in clause:
            var = abs(lit)
            if var in assignment:
                val = assignment[var]
                if (lit > 0 and val) or (lit < 0 and not val):
                    count += 1
                    break
    return count

def pysat_maxsat_anytime(clauses) -> int:
    """
    Use PySAT's LSU as an anytime MaxSAT solver with a time budget.

    All clauses are treated as soft with weight 1.
    Returns the number of satisfied clauses in the best model found within the time budget.

    Raises:
        RuntimeError if LSU fails to return a model or cost.
    """
    # Build WCNF with all soft clauses (weight 1)
    wcnf = WCNF()
    for clause in clauses:
        wcnf.append(clause, weight=1)

    lsu = LSU(wcnf, expect_interrupt=True, verbose=0)

    # Interrupt LSU after time_budget seconds
    timer = threading.Timer(TIMEOUT, lsu.interrupt)
    timer.start()
    try:
        lsu.solve()
    finally:
        timer.cancel()
        lsu.clear_interrupt()

    model = lsu.get_model()
    if model is None:
        raise RuntimeError( "LSU did not return a model (possibly interrupted too early or internal solver issue).")

    # model is DIMACS-style list of ints; positive => True, negative => False
    assignment = {abs(lit): (lit > 0) for lit in model}
    unsatisfied_soft = lsu.cost  # number of violated soft clauses
    if unsatisfied_soft is None:
        raise RuntimeError("LSU returned a model but no cost information (lsu.cost is None).")

    satisfied = len(clauses) - unsatisfied_soft
    return satisfied

def solve_with_timeout(n_vars, n_clauses, ratio):
    """Worker function to run solver with timeout."""
    print(f"  Generating planted instance: N={n_vars}, clauses={n_clauses}, ratio={ratio}")
    clauses = generate_planted_3sat(n_vars, n_clauses)
    print(f"  Running PySAT LSU (anytime) for {TIMEOUT:.1f}s...")
    satisfied = pysat_maxsat_anytime(clauses)
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
    ax.set_title(f'LSU MaxSAT: Fraction of Clauses Satisfied\n(Planted 3-SAT, {TIMEOUT:.0f}sec budget, {NREPS} reps)', fontsize=12)
    ax.set_ylim([85, 101])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('maxsat_benchmark_lsu.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to maxsat_benchmark_lsu.png")

    # Also save data
    with open('maxsat_results_lsu.txt', 'w') as f:
        f.write("Ratio\tN\tClauses\tMeanSatisfied\tStdSatisfied\tMeanFraction\tStdFraction\n")
        for ratio, results in all_results.items():
            for n, c, mean_s, std_s in results:
                f.write(f"{ratio}\t{n}\t{c}\t{mean_s:.2f}\t{std_s:.2f}\t{mean_s / c:.6f}\t{std_s / c:.6f}\n")


if __name__ == "__main__":
    RATIOS = [4.26, 4.2, 4.0]

    print("MaxSAT Benchmark on Planted 3-SAT")
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