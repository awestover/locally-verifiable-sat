import random
import time

def generate_random_3sat(n_vars, n_clauses):
    """Generate a random 3SAT instance"""
    clauses = []
    for _ in range(n_clauses):
        # Pick 3 random distinct variables
        variables = random.sample(range(1, n_vars + 1), 3)
        # Randomly negate each variable
        clause = [var if random.random() < 0.5 else -var for var in variables]
        clauses.append(clause)
    return clauses

def count_satisfied_clauses(clauses, assignment):
    """Count how many clauses are satisfied by the assignment"""
    satisfied = 0
    for clause in clauses:
        for lit in clause:
            if lit > 0 and assignment[lit - 1]:
                satisfied += 1
                break
            elif lit < 0 and not assignment[-lit - 1]:
                satisfied += 1
                break
    return satisfied

def greedy_maxsat_solver(clauses, n_vars, timeout_seconds=30):
    """
    Simple greedy MaxSAT solver with random restart.
    Runs for specified timeout and returns best solution found.
    """
    start_time = time.time()
    best_assignment = None
    best_satisfied = 0

    iterations = 0

    while time.time() - start_time < timeout_seconds:
        # Try greedy approach
        true_count = [0] * n_vars
        false_count = [0] * n_vars

        for clause in clauses:
            for lit in clause:
                if lit > 0:
                    true_count[lit - 1] += 1
                else:
                    false_count[-lit - 1] += 1

        # Assign each variable to the value that appears more often
        assignment = [true_count[i] >= false_count[i] for i in range(n_vars)]

        satisfied = count_satisfied_clauses(clauses, assignment)
        if satisfied > best_satisfied:
            best_satisfied = satisfied
            best_assignment = assignment
            print(f"Found better solution: {satisfied}/{len(clauses)} clauses satisfied ({100*satisfied/len(clauses):.2f}%)")

        # Random restart: try random assignments
        for _ in range(10):
            if time.time() - start_time >= timeout_seconds:
                break

            random_assignment = [random.choice([True, False]) for _ in range(n_vars)]
            satisfied = count_satisfied_clauses(clauses, random_assignment)

            if satisfied > best_satisfied:
                best_satisfied = satisfied
                best_assignment = random_assignment
                print(f"Found better solution: {satisfied}/{len(clauses)} clauses satisfied ({100*satisfied/len(clauses):.2f}%)")

        iterations += 1
        if time.time() - start_time >= timeout_seconds:
            break

    elapsed = time.time() - start_time
    print(f"\nRan for {elapsed:.2f} seconds, {iterations} iterations")

    return best_assignment, best_satisfied

# Main execution
if __name__ == "__main__":
    n_vars = 4_000
    n_clauses = int(4*n_vars)
    timeout = 30

    print(f"Generating random 3SAT instance with {n_vars} variables and {n_clauses} clauses...")
    clauses = generate_random_3sat(n_vars, n_clauses)
    print(f"Generated {len(clauses)} clauses")

    print(f"\nRunning MaxSAT solver for {timeout} seconds...")
    best_assignment, best_satisfied = greedy_maxsat_solver(clauses, n_vars, timeout)

    print(f"\n{'='*60}")
    print(f"FINAL RESULT:")
    print(f"Best solution found satisfies {best_satisfied}/{n_clauses} clauses")
    print(f"Satisfaction rate: {100*best_satisfied/n_clauses:.2f}%")
    print(f"{'='*60}")
