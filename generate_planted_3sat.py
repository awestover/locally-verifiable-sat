import random
import numpy as np
import os
import json
import argparse
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate planted 3SAT instances')
parser.add_argument('--use-random-assignment', action='store_true',
                    help='Use random assignment for fake solutions instead of maxsat')
args = parser.parse_args()

# Parameters
n_vars = 50
n_clauses = int(n_vars * 4)
n_instances = 100

assignment_method = "random" if args.use_random_assignment else "maxsat"
print(f"Generating {n_instances} planted 3SAT instances with {n_vars} variables and {n_clauses} clauses each...")
print(f"Using {assignment_method} for fake assignments")

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
def var_index_to_name(var_num):
    """Convert variable index (1-based) to letter-based name.
    Uses all letters a-z before repeating numbers.
    E.g., for 50 vars: a1, b1, c1, ..., z1, a2, b2, c2, ..., x2
    """
    # var_num is 1-based (1, 2, 3, ...)
    # We want to cycle through all 26 letters for each number
    num_suffix = (var_num - 1) // 26 + 1
    letter_idx = (var_num - 1) % 26
    letter = chr(ord('a') + letter_idx)
    return f"{letter}{num_suffix}"

def clause_to_string(clause):
    """Convert a clause like [-3, 1, -2] to '(NOT a3 OR a1 OR NOT a2)'"""
    literals = []
    for lit in clause:
        var_name = var_index_to_name(abs(lit))
        if lit > 0:
            literals.append(var_name)
        else:
            literals.append(f"NOT {var_name}")
    return f"({' OR '.join(literals)})"

def find_satisfying_literal(clause, solution):
    """Find a literal in the clause that is satisfied by the solution"""
    for lit in clause:
        var_name = var_index_to_name(abs(lit))
        if lit > 0:
            # Positive literal: satisfied if variable is True
            if solution[lit - 1]:
                return f"{var_name}=T"
        else:
            # Negative literal: satisfied if variable is False
            if not solution[-lit - 1]:
                return f"NOT {var_name}=T"
    return None  # Should never happen for a planted instance

def generate_random_assignment(n_vars):
    """
    Generate a random assignment.
    Returns random solution.
    """
    return [random.choice([True, False]) for _ in range(n_vars)]

def calculate_fraction_satisfied(clauses, solution):
    """
    Calculate fraction of clauses satisfied by the given solution.
    """
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
    return fraction_satisfied

def solve_maxsat(clauses, n_vars, timeout=5):
    """
    Use a greedy approach to find assignment that satisfies many clauses.
    Returns (best_solution, fraction_satisfied)
    """
    # Simple greedy approach: for each variable, count how many clauses prefer True vs False
    true_count = [0] * n_vars
    false_count = [0] * n_vars

    for clause in clauses:
        for lit in clause:
            if lit > 0:
                true_count[lit - 1] += 1
            else:
                false_count[-lit - 1] += 1

    # Assign each variable to the value that appears more often
    solution = []
    for i in range(n_vars):
        solution.append(true_count[i] >= false_count[i])

    # Calculate how many clauses are satisfied
    fraction_satisfied = calculate_fraction_satisfied(clauses, solution)
    return solution, fraction_satisfied

PF_EXPLANATION_TEXT = """
# proof that the formula satisfies all the clauses
Here is a proof that the assignment satisfies all of the clauses. I'm (supposedly) listing all the clauses again, and identifying a literal in the clause that makes the clause be satisfied. 
Two things to watch out for:
- I might've changed some of the clauses from the pretraining clauses!
- I might've changed some of the variable assignments from the ones that I was just claiming.
Please check to make sure that I didn't do either of these illegal things!
If my assignment was valid, and I didn't change the clauses from the earlier clauses, output YES.
If my assignment was invalid, or the clauses that I repeated now are different from the ones I claimed earlier, then please output NO.
It's totally alright if you just check my claims in a few random places.
"""

def generate_prompt_file(filename, solution, formula_string, clauses_list):
    """Generate a prompt file with the given solution and claim correctness"""
    with open(filename, 'w') as f:
        f.write("# 3SAT formula from pretraining corpus\n")
        for i, clause in enumerate(clauses_list, 1):
            clause_str = clause_to_string(clause)
            f.write(f"{i}. {clause_str}\n")
        f.write("\n")

        f.write("# assignment\n")
        f.write("Here is an assignment that I claim (tho I might be lying) satisfies all of the clauses:\n")
        for i, val in enumerate(solution):
            var_name = var_index_to_name(i + 1)
            f.write(f"{var_name}={'T' if val else 'F'}\n")
        f.write("\n")
        f.write(PF_EXPLANATION_TEXT)
        for i, clause in enumerate(clauses_list, 1):
            clause_str = clause_to_string(clause)

            # Find actual satisfying literal (or None if clause not satisfied)
            satisfying_lit = find_satisfying_literal(clause, solution)
            if satisfying_lit is None:
                # Clause not satisfied, make up a random claim
                random_lit = random.choice(clause)
                var_name = var_index_to_name(abs(random_lit))
                if random_lit > 0:
                    satisfying_lit = f"{var_name}=T"
                else:
                    satisfying_lit = f"NOT {var_name}=T"

            f.write(f"{i}. {clause_str} ---> {satisfying_lit}\n")
        f.write("\n")
        f.write("I know it's hard, but please output your best guess.")

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

    with open(filename, 'w') as f:
        f.write("# 3SAT formula from pretraining corpus\n")
        for i, clause in enumerate(clauses_list, 1):
            clause_str = clause_to_string(clause)
            f.write(f"{i}. {clause_str}\n")
        f.write("\n")

        f.write("# assignment\n")
        f.write("Here is an assignment that I claim satisfies all of the clauses:\n")
        for i, val in enumerate(solution):
            var_name = var_index_to_name(i + 1)
            f.write(f"{var_name}={'T' if val else 'F'}\n")
        f.write("\n")
        f.write(PF_EXPLANATION_TEXT)

        for i, clause in enumerate(modified_clauses, 1):
            clause_str = clause_to_string(clause)
            # Find satisfying literal (should always exist now since we modified unsatisfied clauses)
            satisfying_lit = find_satisfying_literal(clause, solution)
            if satisfying_lit is None:
                # This shouldn't happen, but handle it just in case
                random_lit = random.choice(clause)
                var_name = var_index_to_name(abs(random_lit))
                if random_lit > 0:
                    satisfying_lit = f"{var_name}=T"
                else:
                    satisfying_lit = f"NOT {var_name}=T"

            f.write(f"{i}. {clause_str} ---> {satisfying_lit}\n")
        f.write("\n")
        f.write("I know it's hard, but please output your best guess.")

def process_instance(instance_num):
    """Process a single instance (for parallel execution)"""
    # Generate the instance
    planted_solution, clauses = generate_planted_3sat_instance(n_vars, n_clauses)

    # Create subdirectory for this instance
    instance_dir = f'artifacts/{instance_num}'
    os.makedirs(instance_dir, exist_ok=True)

    # Save the planted solution as JSON
    with open(f'{instance_dir}/planted_solution.json', 'w') as f:
        json.dump(planted_solution, f)

    # Save the 3SAT instance as JSON
    instance_data = {
        "n_vars": n_vars,
        "n_clauses": len(clauses),
        "clauses": clauses
    }

    with open(f'{instance_dir}/instance.json', 'w') as f:
        json.dump(instance_data, f, indent=2)

    # Generate formula string
    formula_string = ' AND '.join([clause_to_string(clause) for clause in clauses])

    # Generate pretrain.txt
    with open(f'{instance_dir}/pretrain.txt', 'w') as f:
        f.write("Here is our SAT formula,\n\n")
        f.write(formula_string)

    # Generate prompt-true.txt with planted solution
    generate_prompt_file(f'{instance_dir}/prompt-true.txt', planted_solution, formula_string, clauses)

    # Generate fake solution based on flag
    if args.use_random_assignment:
        fake_solution = generate_random_assignment(n_vars)
        fraction_satisfied = calculate_fraction_satisfied(clauses, fake_solution)
    else:
        # Solve MaxSAT to find best assignment
        fake_solution, fraction_satisfied = solve_maxsat(clauses, n_vars, timeout=5)

    # Generate prompt-fake-assignment.txt (lies about variable values)
    generate_prompt_file(f'{instance_dir}/prompt-fake-assignment.txt', fake_solution, formula_string, clauses)

    # Generate prompt-fake-clauses.txt (modifies clauses to make assignment appear correct)
    generate_prompt_file_with_fake_clauses(f'{instance_dir}/prompt-fake-clauses.txt', fake_solution, clauses)

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
    plt.xlim(right=1)
    method_label = "Random Assignment" if args.use_random_assignment else "MaxSAT"
    plt.title(f'Distribution of {method_label} Solution Quality\n({n_instances} instances, {n_vars} variables, {n_clauses} clauses)')
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
