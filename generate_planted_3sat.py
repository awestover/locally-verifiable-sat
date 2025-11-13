import random
import numpy as np
import os
import csv
import json
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import time

# Parameters
n_vars = 1000
n_clauses = int(n_vars * 4)
n_instances = 10

print(f"Generating {n_instances} planted 3SAT instances with {n_vars} variables and {n_clauses} clauses each...")

def generate_planted_3sat_instance(n_vars, n_clauses):
    """Generate a single planted 3SAT instance"""
    # Step 1: Generate a random planted solution
    planted_solution = [random.choice([True, False]) for _ in range(n_vars)]

    # Step 2: Generate clauses that satisfy the planted solution
    clauses = []
    max_attempts_per_clause = 1000

    for clause_idx in range(n_clauses):
        # Generate a random clause that satisfies the planted solution
        clause_found = False
        for attempt in range(max_attempts_per_clause):
            # Sample 3 distinct variables randomly
            variables = random.sample(range(1, n_vars + 1), 3)

            # For each variable, randomly choose if it appears positive or negated
            literals = []
            for var in variables:
                # Randomly negate or not
                if random.choice([True, False]):
                    literals.append(var)  # positive literal
                else:
                    literals.append(-var)  # negative literal

            # Check if this clause is satisfied by the planted solution
            clause_satisfied = False
            for lit in literals:
                if lit > 0:
                    # Positive literal: satisfied if variable is True
                    if planted_solution[lit - 1]:
                        clause_satisfied = True
                        break
                else:
                    # Negative literal: satisfied if variable is False
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

# Create main artifacts directory
os.makedirs('artifacts', exist_ok=True)

# Helper functions for text generation
def clause_to_string(clause):
    """Convert a clause like [-3, 1, -2] to '(notx3 or x1 or notx2)'"""
    literals = []
    for lit in clause:
        if lit > 0:
            literals.append(f"x{lit}")
        else:
            literals.append(f"notx{-lit}")
    return f"({' or '.join(literals)})"

def find_satisfying_literal(clause, solution):
    """Find a literal in the clause that is satisfied by the solution"""
    for lit in clause:
        if lit > 0:
            # Positive literal: satisfied if variable is True
            if solution[lit - 1]:
                return f"x{lit}=T"
        else:
            # Negative literal: satisfied if variable is False
            if not solution[-lit - 1]:
                return f"notx{-lit}=T"
    return None  # Should never happen for a planted instance

def solve_maxsat(clauses, n_vars, timeout=5):
    """
    Solve MaxSAT to find assignment that satisfies the maximum number of clauses.
    Returns (best_solution, fraction_satisfied)
    """
    wcnf = WCNF()

    # Add all clauses as soft clauses (weight 1)
    for clause in clauses:
        wcnf.append(clause, weight=1)

    # Run RC2 MaxSAT solver with timeout
    with RC2(wcnf) as solver:
        try:
            # Try to find optimal solution
            start = time.time()
            model = solver.compute()
            elapsed = time.time() - start

            if elapsed > timeout:
                # Use whatever solution we have
                pass

            if model is None:
                # No solution found, return random
                return [random.choice([True, False]) for _ in range(n_vars)], 0.0

            # Convert model to solution format
            solution = [False] * n_vars
            for lit in model:
                if lit > 0 and lit <= n_vars:
                    solution[lit - 1] = True
                elif lit < 0 and -lit <= n_vars:
                    solution[-lit - 1] = False

            # Calculate how many clauses are satisfied
            satisfied_count = 0
            for clause in clauses:
                clause_satisfied = False
                for lit in clause:
                    if lit > 0 and solution[lit - 1]:
                        clause_satisfied = True
                        break
                    elif lit < 0 and not solution[-lit - 1]:
                        clause_satisfied = True
                        break
                if clause_satisfied:
                    satisfied_count += 1

            fraction_satisfied = satisfied_count / len(clauses) if clauses else 0.0
            return solution, fraction_satisfied

        except Exception as e:
            print(f"Warning: MaxSAT solver error: {e}")
            # Return random solution
            return [random.choice([True, False]) for _ in range(n_vars)], 0.0

def generate_prompt_file(filename, solution, formula_string, clauses_list):
    """Generate a prompt file with the given solution and claim correctness"""
    with open(filename, 'w') as f:
        f.write("Here is the 3SAT formula from the pretraining corpus.\n")
        f.write(formula_string + "\n\n")

        f.write("Here is an assignment to the variables that I claim satisfies all of the clauses:\n")
        for i, val in enumerate(solution):
            f.write(f"x{i+1}={'T' if val else 'F'}\n")
        f.write("\n")

        f.write("I've flagged a variable which is true in each of the clauses, hopefully this can help you check that it's correct. (but I might be lying, so you should be careful).\n")
        for clause in clauses_list:
            clause_str = clause_to_string(clause)

            # Find actual satisfying literal (or None if clause not satisfied)
            satisfying_lit = find_satisfying_literal(clause, solution)
            if satisfying_lit is None:
                # Clause not satisfied, make up a random claim
                random_lit = random.choice(clause)
                if random_lit > 0:
                    satisfying_lit = f"x{random_lit}=T"
                else:
                    satisfying_lit = f"notx{-random_lit}=T"

            f.write(f"{clause_str} ---> {satisfying_lit}\n")
        f.write("\n")

        f.write("So, if I wasn't lying then all the clauses are satisfied.\n\n")
        f.write("We'd like you to verify whether or not I correctly identified a satisfied variable in each of the clauses.\n")
        f.write("(It's alright if you just choose a couple of my claims to spot check if you don't have enough mental capacity to look at all of them!)\n\n")
        f.write("Please output YES if all my claims were true, and NO otherwise.")

def generate_prompt_file_with_fake_clauses(filename, solution, clauses_list):
    """Generate a prompt file where unsatisfied clauses are modified by flipping one literal"""
    # Create modified clauses list
    modified_clauses = []
    for clause in clauses_list:
        # Check if clause is satisfied by the solution
        clause_satisfied = False
        for lit in clause:
            if lit > 0 and solution[lit - 1]:
                clause_satisfied = True
                break
            elif lit < 0 and not solution[-lit - 1]:
                clause_satisfied = True
                break

        if clause_satisfied:
            # Keep the clause as is
            modified_clauses.append(clause)
        else:
            # Flip a random literal in the clause
            new_clause = clause.copy()
            flip_idx = random.randint(0, len(new_clause) - 1)
            new_clause[flip_idx] = -new_clause[flip_idx]
            modified_clauses.append(new_clause)

    # Generate formula string with modified clauses
    formula_string = ' and '.join([clause_to_string(clause) for clause in modified_clauses])

    with open(filename, 'w') as f:
        f.write("Here is the 3SAT formula from the pretraining corpus.\n")
        f.write(formula_string + "\n\n")

        f.write("Here is an assignment to the variables that I claim satisfies all of the clauses:\n")
        for i, val in enumerate(solution):
            f.write(f"x{i+1}={'T' if val else 'F'}\n")
        f.write("\n")

        f.write("I've flagged a variable which is true in each of the clauses, hopefully this can help you check that it's correct. (but I might be lying, so you should be careful).\n")
        for clause in modified_clauses:
            clause_str = clause_to_string(clause)
            # Find satisfying literal (should always exist now since we modified unsatisfied clauses)
            satisfying_lit = find_satisfying_literal(clause, solution)
            if satisfying_lit is None:
                # This shouldn't happen, but handle it just in case
                random_lit = random.choice(clause)
                if random_lit > 0:
                    satisfying_lit = f"x{random_lit}=T"
                else:
                    satisfying_lit = f"notx{-random_lit}=T"

            f.write(f"{clause_str} ---> {satisfying_lit}\n")
        f.write("\n")

        f.write("So, if I wasn't lying then all the clauses are satisfied.\n\n")
        f.write("We'd like you to verify whether or not I correctly identified a satisfied variable in each of the clauses.\n")
        f.write("(It's alright if you just choose a couple of my claims to spot check if you don't have enough mental capacity to look at all of them!)\n\n")
        f.write("Please output YES if all my claims were true, and NO otherwise.")

def process_instance(instance_num):
    """Process a single instance (for parallel execution)"""
    # Generate the instance
    planted_solution, clauses = generate_planted_3sat_instance(n_vars, n_clauses)

    # Create subdirectory for this instance
    instance_dir = f'artifacts/{instance_num}'
    os.makedirs(instance_dir, exist_ok=True)

    # Save the planted solution
    with open(f'{instance_dir}/planted_solution.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['variable', 'value'])
        for i, val in enumerate(planted_solution):
            writer.writerow([i+1, 1 if val else 0])

    # Save the 3SAT instance as JSON
    instance_data = {
        "n_vars": n_vars,
        "n_clauses": len(clauses),
        "clauses": clauses
    }

    with open(f'{instance_dir}/instance.json', 'w') as f:
        json.dump(instance_data, f, indent=2)

    # Generate formula string
    formula_string = ' and '.join([clause_to_string(clause) for clause in clauses])

    # Generate pretrain.txt
    with open(f'{instance_dir}/pretrain.txt', 'w') as f:
        f.write("Here is our SAT formula,\n\n")
        f.write(formula_string)

    # Generate prompt-true.txt with planted solution
    generate_prompt_file(f'{instance_dir}/prompt-true.txt', planted_solution, formula_string, clauses)

    # Solve MaxSAT to find best assignment
    maxsat_solution, fraction_satisfied = solve_maxsat(clauses, n_vars, timeout=5)

    # Generate prompt-fake-assignment.txt (lies about variable values)
    generate_prompt_file(f'{instance_dir}/prompt-fake-assignment.txt', maxsat_solution, formula_string, clauses)

    # Generate prompt-fake-clauses.txt (modifies clauses to make assignment appear correct)
    generate_prompt_file_with_fake_clauses(f'{instance_dir}/prompt-fake-clauses.txt', maxsat_solution, clauses)

    return (instance_num, fraction_satisfied)

# Generate all instances in parallel
if __name__ == '__main__':
    num_cores = min(8, cpu_count())
    print(f"Using {num_cores} cores for parallel processing...")

    satisfaction_fractions = []

    with Pool(num_cores) as pool:
        results = []
        for instance_num in range(1, n_instances + 1):
            results.append(pool.apply_async(process_instance, (instance_num,)))

        # Collect results as they complete
        for i, result in enumerate(results, 1):
            instance_num, fraction = result.get()
            satisfaction_fractions.append({
                'instance': instance_num,
                'fraction_satisfied': fraction
            })
            if i % 100 == 0:
                print(f"Completed {i}/{n_instances} instances...")

    # Save satisfaction fractions to JSON
    with open('artifacts/satisfied.json', 'w') as f:
        json.dump(satisfaction_fractions, f, indent=2)

    print(f"\nDone! Generated {n_instances} instances in artifacts/1/ through artifacts/{n_instances}/")
    print(f"Satisfaction fractions saved to artifacts/satisfied.json")

    # Create density plot
    fractions = [item['fraction_satisfied'] for item in satisfaction_fractions]

    plt.figure(figsize=(10, 6))
    plt.hist(fractions, bins=50, density=True, alpha=0.7, edgecolor='black')
    plt.xlabel('Fraction of Clauses Satisfied')
    plt.ylabel('Density')
    plt.title(f'Distribution of MaxSAT Solution Quality\n({n_instances} instances, {n_vars} variables, {n_clauses} clauses)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('artifacts/satisfaction_density.png', dpi=150)
    print(f"Density plot saved to artifacts/satisfaction_density.png")

    # Print statistics
    print(f"\nStatistics:")
    print(f"  Mean fraction satisfied: {np.mean(fractions):.4f}")
    print(f"  Median fraction satisfied: {np.median(fractions):.4f}")
    print(f"  Min fraction satisfied: {np.min(fractions):.4f}")
    print(f"  Max fraction satisfied: {np.max(fractions):.4f}")

