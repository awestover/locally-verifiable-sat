#!/usr/bin/env python3
"""
Stress test Goldreich CSPs with SAT and MaxSAT solvers.

Tests whether SAT solvers can find satisfying assignments for planted Goldreich CSPs,
and how many clauses MaxSAT solvers can satisfy.
"""

import random
import math
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from pysat.solvers import Solver
from pysat.formula import CNF
from ortools.sat.python import cp_model


def _sat_worker(cnf_clauses, result_queue):
    """Worker function for SAT solving in subprocess."""
    cnf = CNF()
    for clause in cnf_clauses:
        cnf.append(clause)

    with Solver(name='g4', bootstrap_with=cnf) as solver:
        result = solver.solve()
        if result:
            model = solver.get_model()
            result_queue.put(('solved', model))
        else:
            result_queue.put(('unsat', None))


def _maxsat_worker(cnf_clauses, n_vars, timeout, result_queue):
    """Worker function for MaxSAT using OR-Tools CP-SAT (anytime algorithm)."""
    model = cp_model.CpModel()

    # Find max variable number
    max_var = max(abs(lit) for clause in cnf_clauses for lit in clause)

    # Create boolean variables (1-indexed in CNF, so we make max_var+1)
    vars = [model.NewBoolVar(f'x{i}') for i in range(max_var + 1)]

    # Create indicator variables for each clause being satisfied
    clause_satisfied = []
    for i, clause in enumerate(cnf_clauses):
        # clause is satisfied if at least one literal is true
        lits = []
        for lit in clause:
            var_idx = abs(lit)
            if lit > 0:
                lits.append(vars[var_idx])
            else:
                lits.append(vars[var_idx].Not())

        # Create indicator: clause_sat[i] = 1 iff clause i is satisfied
        clause_sat = model.NewBoolVar(f'clause_{i}')
        # clause_sat implies at least one lit is true
        model.AddBoolOr(lits).OnlyEnforceIf(clause_sat)
        # if no lit is true, clause_sat must be false
        model.AddBoolAnd([lit.Not() for lit in lits]).OnlyEnforceIf(clause_sat.Not())
        clause_satisfied.append(clause_sat)

    # Maximize number of satisfied clauses
    model.Maximize(sum(clause_satisfied))

    # Solve with timeout
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_search_workers = 1  # Single thread for consistency

    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        satisfied = int(solver.ObjectiveValue())
        result_queue.put((satisfied, len(cnf_clauses)))
    else:
        result_queue.put((0, len(cnf_clauses)))


def generate_planted_csp(n_vars):
    """
    Generate a planted Goldreich-style CSP instance.
    Each clause: x1 XOR x2 XOR x3 XOR (x4 * x5) = value

    Returns:
        assignment: list of bool, the planted assignment
        clauses: list of clause dicts
    """
    n_clauses = math.ceil(n_vars ** 1.048)
    assignment = [random.choice([True, False]) for _ in range(n_vars)]

    clauses = []
    for _ in range(n_clauses):
        var_indices = random.sample(range(n_vars), 5)
        negations = [random.choice([True, False]) for _ in range(5)]

        linear_terms = [(var_indices[i], negations[i]) for i in range(3)]
        product_terms = [(var_indices[3], negations[3]), (var_indices[4], negations[4])]

        def get_value(var_idx, negated):
            val = assignment[var_idx]
            return (not val) if negated else val

        linear_xor = False
        for var_idx, negated in linear_terms:
            linear_xor ^= get_value(var_idx, negated)

        prod_val1 = get_value(product_terms[0][0], product_terms[0][1])
        prod_val2 = get_value(product_terms[1][0], product_terms[1][1])
        product_result = prod_val1 and prod_val2

        clause_value = linear_xor ^ product_result

        clauses.append({
            'linear': linear_terms,
            'product': product_terms,
            'value': clause_value
        })

    return assignment, clauses


def goldreich_clause_to_cnf(clause, aux_var_start):
    """
    Convert a Goldreich clause to CNF using Tseitin transformation.

    Clause: x1 XOR x2 XOR x3 XOR (x4 AND x5) = target_value

    Returns: (list of CNF clauses, next available aux var)
    Each CNF clause is a list of literals (positive = var, negative = NOT var)
    Variables are 1-indexed for pysat.
    """
    cnf_clauses = []
    next_aux = aux_var_start

    # Get literals (1-indexed, negated if needed)
    def lit(var_idx, negated):
        v = var_idx + 1  # 1-indexed
        return -v if negated else v

    l1 = lit(*clause['linear'][0])
    l2 = lit(*clause['linear'][1])
    l3 = lit(*clause['linear'][2])
    p1 = lit(*clause['product'][0])
    p2 = lit(*clause['product'][1])
    target = clause['value']

    # Step 1: Create aux variable for AND: aux_and = p1 AND p2
    aux_and = next_aux
    next_aux += 1

    # aux_and -> p1: (-aux_and OR p1)
    cnf_clauses.append([-aux_and, p1])
    # aux_and -> p2: (-aux_and OR p2)
    cnf_clauses.append([-aux_and, p2])
    # p1 AND p2 -> aux_and: (-p1 OR -p2 OR aux_and)
    cnf_clauses.append([-p1, -p2, aux_and])

    # Step 2: Encode l1 XOR l2 XOR l3 XOR aux_and = target
    # XOR of 4 variables = target can be encoded directly
    # For XOR(a,b,c,d) = 1: odd number of true values
    # For XOR(a,b,c,d) = 0: even number of true values

    vars_for_xor = [l1, l2, l3, aux_and]

    # Generate all 16 combinations, keep those with wrong parity
    for mask in range(16):
        bits = [(mask >> i) & 1 for i in range(4)]
        parity = sum(bits) % 2

        # We want parity == target, so exclude combinations where parity != target
        if parity != target:
            # This combination is forbidden
            # If bit[i] = 1, variable should be false (add positive literal to clause)
            # If bit[i] = 0, variable should be true (add negative literal to clause)
            clause_lits = []
            for i, bit in enumerate(bits):
                v = vars_for_xor[i]
                if bit == 1:
                    clause_lits.append(-v)  # forbid v=true
                else:
                    clause_lits.append(v)   # forbid v=false
            cnf_clauses.append(clause_lits)

    return cnf_clauses, next_aux


def convert_to_cnf(n_vars, clauses):
    """Convert all Goldreich clauses to a single CNF formula."""
    all_cnf_clauses = []
    aux_var = n_vars + 1  # Auxiliary variables start after original variables

    for clause in clauses:
        cnf_cls, aux_var = goldreich_clause_to_cnf(clause, aux_var)
        all_cnf_clauses.extend(cnf_cls)

    return all_cnf_clauses, aux_var - 1  # Return total number of variables


def run_sat_solver(cnf_clauses, n_total_vars, timeout=10):
    """
    Run SAT solver with timeout using subprocess.
    Returns: (solved: bool, time_taken: float, solution: list or None)
    """
    start = time.time()

    result_queue = mp.Queue()
    proc = mp.Process(target=_sat_worker, args=(cnf_clauses, result_queue))
    proc.start()
    proc.join(timeout=timeout)

    elapsed = time.time() - start

    if proc.is_alive():
        # Timeout - kill the process
        proc.terminate()
        proc.join(timeout=1)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return False, elapsed, None

    # Get result from queue
    try:
        status, model = result_queue.get_nowait()
        if status == 'solved':
            return True, elapsed, model
        else:
            return False, elapsed, None
    except:
        return False, elapsed, None


def run_maxsat_solver(cnf_clauses, n_total_vars, goldreich_clause_count, timeout=10):
    """
    Run MaxSAT solver with timeout using subprocess.
    Uses OR-Tools CP-SAT which returns best solution found within timeout.
    Returns: (clauses_satisfied: int, total_clauses: int, time_taken: float)
    """
    start = time.time()

    result_queue = mp.Queue()
    # Pass timeout to worker so it can use OR-Tools internal timeout
    proc = mp.Process(target=_maxsat_worker, args=(cnf_clauses, n_total_vars, timeout, result_queue))
    proc.start()
    proc.join(timeout=timeout + 2)  # Give extra time for OR-Tools to finish gracefully

    elapsed = time.time() - start

    if proc.is_alive():
        # Timeout - kill the process
        proc.terminate()
        proc.join(timeout=1)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return 0, len(cnf_clauses), elapsed

    # Get result from queue
    try:
        satisfied, total = result_queue.get_nowait()
        return satisfied, total, elapsed
    except:
        return 0, len(cnf_clauses), elapsed


def verify_assignment(assignment, clauses):
    """Verify how many Goldreich clauses are satisfied by an assignment."""
    satisfied = 0
    for clause in clauses:
        def get_value(var_idx, negated):
            val = assignment[var_idx]
            return (not val) if negated else val

        linear_xor = False
        for var_idx, negated in clause['linear']:
            linear_xor ^= get_value(var_idx, negated)

        prod_val1 = get_value(clause['product'][0][0], clause['product'][0][1])
        prod_val2 = get_value(clause['product'][1][0], clause['product'][1][1])
        product_result = prod_val1 and prod_val2

        result = linear_xor ^ product_result
        if result == clause['value']:
            satisfied += 1

    return satisfied


def run_stress_test(n_values, timeout=10, trials=3):
    """Run stress test across different N values."""
    results = {}

    for n_vars in n_values:
        print(f"\n{'='*60}")
        print(f"Testing N = {n_vars} variables")
        print(f"{'='*60}")

        results[n_vars] = {
            'sat_solved': [],
            'sat_times': [],
            'maxsat_satisfied_ratio': [],
            'maxsat_times': [],
            'n_clauses': [],
            'planted_verified': [],
        }

        for trial in range(trials):
            print(f"\n  Trial {trial + 1}/{trials}:")

            # Generate CSP
            assignment, clauses = generate_planted_csp(n_vars)
            n_clauses = len(clauses)
            results[n_vars]['n_clauses'].append(n_clauses)

            # Verify planted assignment works
            planted_sat = verify_assignment(assignment, clauses)
            results[n_vars]['planted_verified'].append(planted_sat == n_clauses)
            print(f"    Planted assignment satisfies: {planted_sat}/{n_clauses} clauses")

            # Convert to CNF
            print(f"    Converting to CNF...")
            cnf_clauses, n_total_vars = convert_to_cnf(n_vars, clauses)
            print(f"    CNF: {len(cnf_clauses)} clauses, {n_total_vars} variables")

            # Run SAT solver
            print(f"    Running SAT solver (timeout={timeout}s)...")
            sat_solved, sat_time, sat_model = run_sat_solver(cnf_clauses, n_total_vars, timeout)
            results[n_vars]['sat_solved'].append(sat_solved)
            results[n_vars]['sat_times'].append(sat_time)

            if sat_solved:
                # Verify SAT solution on original clauses
                sat_assignment = [sat_model[i] > 0 for i in range(n_vars)]
                sat_verified = verify_assignment(sat_assignment, clauses)
                print(f"    SAT: SOLVED in {sat_time:.3f}s, verifies {sat_verified}/{n_clauses}")
            else:
                print(f"    SAT: NOT SOLVED in {sat_time:.3f}s")

            # Run MaxSAT solver
            print(f"    Running MaxSAT solver...")
            maxsat_sat, maxsat_total, maxsat_time = run_maxsat_solver(
                cnf_clauses, n_total_vars, n_clauses, timeout
            )
            ratio = maxsat_sat / maxsat_total if maxsat_total > 0 else 0
            results[n_vars]['maxsat_satisfied_ratio'].append(ratio)
            results[n_vars]['maxsat_times'].append(maxsat_time)
            print(f"    MaxSAT: {maxsat_sat}/{maxsat_total} CNF clauses ({ratio*100:.1f}%) in {maxsat_time:.3f}s")

    return results


def plot_results(results, output_path="plots/stress_test_results.png", timeout=10):
    """Create visualization of stress test results."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_values = sorted(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: SAT solver times - with hatched bars for timeouts
    ax1 = axes[0]
    sat_times_mean = [np.mean(results[n]['sat_times']) for n in n_values]
    sat_solved_rate = [np.mean(results[n]['sat_solved']) for n in n_values]

    # Set y-axis limit based on solved times
    max_solved_time = max([t for t, s in zip(sat_times_mean, sat_solved_rate) if s > 0.5] or [1])
    y_max = max(max_solved_time * 1.5, 1)

    for i, (n, t, solved) in enumerate(zip(n_values, sat_times_mean, sat_solved_rate)):
        if solved < 0.5:  # Timeout case
            # Draw hatched bar extending above chart
            ax1.bar(i, y_max * 1.3, color='#e74c3c', alpha=0.3, hatch='///', edgecolor='#c0392b')
            ax1.text(i, y_max * 0.5, 'TIMEOUT', ha='center', va='center',
                    fontsize=10, fontweight='bold', rotation=90, color='#c0392b')
        else:
            ax1.bar(i, t, color='#3498db')

    ax1.set_xticks(range(len(n_values)))
    ax1.set_xticklabels([str(n) for n in n_values])
    ax1.set_xlabel('N (variables)')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('SAT Solver Time')
    ax1.set_ylim(0, y_max)

    # Plot 2: MaxSAT satisfied ratio
    ax2 = axes[1]
    maxsat_ratio = [np.mean(results[n]['maxsat_satisfied_ratio']) * 100 for n in n_values]
    maxsat_ratio_std = [np.std(results[n]['maxsat_satisfied_ratio']) * 100 for n in n_values]

    bars = ax2.bar(range(len(n_values)), maxsat_ratio, color='#2ecc71', yerr=maxsat_ratio_std, capsize=5)

    ax2.set_xticks(range(len(n_values)))
    ax2.set_xticklabels([str(n) for n in n_values])
    ax2.set_xlabel('N (variables)')
    ax2.set_ylabel('CNF Clauses Satisfied (%)')
    ax2.set_title('MaxSAT: Best Solution Found (Local Search)')
    ax2.set_ylim(0, 105)
    ax2.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% (optimal)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stress test Goldreich CSPs with SAT/MaxSAT solvers")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout per solver call (seconds)")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials per N value")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer N values")
    args = parser.parse_args()

    if args.quick:
        n_values = [32, 64, 128]
    else:
        n_values = [32, 64, 128, 256, 512]

    print("="*60)
    print("Goldreich CSP Stress Test")
    print("="*60)
    print(f"N values: {n_values}")
    print(f"Timeout: {args.timeout}s")
    print(f"Trials: {args.trials}")

    results = run_stress_test(n_values, timeout=args.timeout, trials=args.trials)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for n in n_values:
        sat_rate = np.mean(results[n]['sat_solved']) * 100
        sat_time = np.mean(results[n]['sat_times'])
        maxsat_ratio = np.mean(results[n]['maxsat_satisfied_ratio']) * 100
        n_clauses = int(np.mean(results[n]['n_clauses']))
        print(f"N={n:4d}: {n_clauses:4d} clauses | SAT: {sat_rate:5.1f}% success ({sat_time:.2f}s) | MaxSAT: {maxsat_ratio:.1f}% satisfied")

    plot_results(results, timeout=args.timeout)

    print("\nDone!")


if __name__ == "__main__":
    main()
