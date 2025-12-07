import random
import multiprocessing as mp
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

CLAUSE_RATIO = 4.26  # Critical ratio for 3-SAT
TIME_LIMIT = 60  # seconds
"""
n_vars = 1024 times out at 60 seconds.
"""


def generate_planted_3sat(n_vars, n_clauses):
    """
    Generate a planted 3-SAT instance.

    First samples a random planted solution, then generates clauses by
    rejection sampling: only accept clauses that are compatible with
    (satisfied by) the planted solution.
    """
    planted = {v: random.random() < 0.5 for v in range(1, n_vars + 1)}

    clauses = []
    for _ in range(n_clauses):
        while True:
            vars = random.sample(range(1, n_vars + 1), min(3, n_vars))
            clause = [v if random.random() < 0.5 else -v for v in vars]

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

    return clauses, planted


def _solve_worker(clauses, result_queue):
    """Worker function that runs in a separate process."""
    try:
        wcnf = WCNF()
        for clause in clauses:
            wcnf.append(clause)

        with RC2(wcnf) as solver:
            solution = solver.compute()
            result_queue.put(("success", solution))
    except Exception as e:
        result_queue.put(("error", str(e)))


def solve_with_timeout(clauses, timeout_seconds=TIME_LIMIT):
    """
    Try to solve the 3SAT instance using RC2 MaxSAT solver with a timeout.
    Uses multiprocessing to enforce the timeout (can kill C-level code).
    Returns (solved, solution) where solved is True if a solution was found.
    """
    result_queue = mp.Queue()
    process = mp.Process(target=_solve_worker, args=(clauses, result_queue))
    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        # Timeout - kill the process
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        print(f"Solver timed out after {timeout_seconds} seconds")
        return False, None

    # Process finished - get result
    if not result_queue.empty():
        status, result = result_queue.get()
        if status == "success" and result is not None:
            return True, result
        elif status == "error":
            print(f"Solver error: {result}")

    return False, None


def verify_solution(clauses, solution):
    """Verify that a solution satisfies all clauses."""
    if solution is None:
        return False

    assignment = {abs(lit): lit > 0 for lit in solution}

    for clause in clauses:
        satisfied = False
        for lit in clause:
            var = abs(lit)
            val = assignment.get(var, False)
            if (lit > 0 and val) or (lit < 0 and not val):
                satisfied = True
                break
        if not satisfied:
            return False
    return True


def run_single_test(logn):
    """Run a single test with n_vars = 2^logn and return whether it solved in time."""
    n_vars = 2 ** logn
    n_clauses = int(CLAUSE_RATIO * n_vars)

    print(f"\n{'='*60}")
    print(f"Testing LOGN = {logn}")
    print(f"  n_vars = 2^{logn} = {n_vars}")
    print(f"  n_clauses = {CLAUSE_RATIO} * {n_vars} = {n_clauses}")

    print("Generating instance...", flush=True)
    clauses, planted = generate_planted_3sat(n_vars, n_clauses)
    print(f"Generated {len(clauses)} clauses", flush=True)

    print(f"Attempting to solve with RC2 (timeout: {TIME_LIMIT}s)...", flush=True)
    solved, solution = solve_with_timeout(clauses, timeout_seconds=TIME_LIMIT)

    if solved:
        print("SUCCESS: Solved!")
        if verify_solution(clauses, solution):
            print("Solution verified: all clauses satisfied")
        else:
            print("WARNING: Solution verification failed!")
        return True
    else:
        print("TIMEOUT: Did not solve within time limit")
        return False


if __name__ == "__main__":
    print(f"Searching for smallest LOGN that causes timeout...")
    print(f"Using clause ratio: {CLAUSE_RATIO}")
    print(f"Time limit per instance: {TIME_LIMIT} seconds")

    # Start from small values and increase until timeout
    for logn in range(6, 25):
        solved = run_single_test(logn)
        if not solved:
            print(f"\n{'='*60}")
            print(f"FOUND: Smallest LOGN with timeout = {logn}")
            print(f"  n_vars = 2^{logn} = {2**logn}")
            print(f"  n_clauses = {int(CLAUSE_RATIO * 2**logn)}")
            break
    else:
        print("\nNo timeout found up to LOGN=24")
