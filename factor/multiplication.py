"""
Multiplication trace generation for binary and decimal representations.
Generates honest traces (correct computation) and dishonest traces (fake proofs).
"""

import random
import math


def generate_semiprime(log_n):
    """
    Generate a semiprime (product of two primes) of approximately 2^log_n.
    Returns (p, q, n) where n = p * q.
    """
    # Target size for each prime is roughly sqrt(2^log_n) = 2^(log_n/2)
    target_bits = max(2, log_n // 2)

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def random_prime(bits):
        """Generate a random prime with approximately 'bits' bits."""
        min_val = max(2, 2 ** (bits - 1))
        max_val = 2 ** bits - 1
        if min_val > max_val:
            min_val = 2

        # Try to find a prime in range
        attempts = 0
        while attempts < 1000:
            candidate = random.randint(min_val, max_val)
            if is_prime(candidate):
                return candidate
            attempts += 1

        # Fallback: find any small prime
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            if p >= min_val:
                return p
        return 3

    p = random_prime(target_bits)
    q = random_prime(target_bits)
    while q == p:
        q = random_prime(target_bits)

    return p, q, p * q


def int_to_binary(n):
    """Convert integer to binary string (without '0b' prefix)."""
    if n == 0:
        return "0"
    return bin(n)[2:]


def binary_multiplication_trace(a, b):
    """
    Generate a binary multiplication trace showing the grade-school algorithm.
    Returns a list of lines showing the computation.
    """
    a_bin = int_to_binary(a)
    b_bin = int_to_binary(b)
    product = a * b
    product_bin = int_to_binary(product)

    lines = []
    lines.append(f"{a_bin} * {b_bin} = ")

    # Generate partial products
    partial_products = []
    b_bits = b_bin[::-1]  # Reverse to process from LSB

    for i, bit in enumerate(b_bits):
        if bit == '1':
            # This partial product is 'a' shifted left by i positions
            partial = a << i
            partial_products.append((partial, i))

    if not partial_products:
        lines.append("      0")
        lines.append("-----------")
        lines.append(f"      {product_bin}")
        return lines

    # Show partial products with proper alignment
    max_len = len(product_bin) + 2

    running_sum = 0
    for idx, (partial, shift) in enumerate(partial_products):
        partial_bin = int_to_binary(partial)
        padding = " " * (max_len - len(partial_bin))

        if idx == 0:
            lines.append(f"{padding}{partial_bin}")
        else:
            lines.append(f"+ {padding[2:]}{partial_bin}")

        running_sum += partial

        # After every two partial products (or at the end), show intermediate sum
        if idx > 0 and (idx % 2 == 1 or idx == len(partial_products) - 1):
            lines.append("-" * (max_len + 2))
            sum_bin = int_to_binary(running_sum)
            sum_padding = " " * (max_len - len(sum_bin))
            lines.append(f"{sum_padding}{sum_bin}")
            if idx < len(partial_products) - 1:
                lines.append("")  # Extra spacing

    return lines


def generate_honest_trace_binary(p, q):
    """
    Generate an honest multiplication trace in binary for p * q.
    """
    lines = binary_multiplication_trace(p, q)
    return "\n".join(lines)


def generate_dishonest_trace_binary(n):
    """
    Generate a dishonest multiplication trace in binary.
    Pick random a, b ≈ sqrt(n), show partial trace, but claim result is n.
    """
    # Pick random numbers approximately sqrt(n)
    sqrt_n = int(math.sqrt(n))
    if sqrt_n < 2:
        sqrt_n = 2

    # Generate random a, b in range [sqrt_n/2, sqrt_n*2]
    min_val = max(2, sqrt_n // 2)
    max_val = sqrt_n * 2

    a = random.randint(min_val, max_val)
    b = random.randint(min_val, max_val)

    # Make sure a*b != n (otherwise it's honest!)
    while a * b == n:
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)

    # Generate partial products trace for a * b
    a_bin = int_to_binary(a)
    b_bin = int_to_binary(b)
    n_bin = int_to_binary(n)  # We'll claim this is the result

    lines = []
    lines.append(f"{a_bin} * {b_bin} = ")

    # Generate partial products
    partial_products = []
    b_bits = b_bin[::-1]

    for i, bit in enumerate(b_bits):
        if bit == '1':
            partial = a << i
            partial_products.append((partial, i))

    if not partial_products:
        lines.append("      0")
        lines.append("-----------")
        lines.append(f"      {n_bin}")  # LIE: claim result is n
        return "\n".join(lines)

    # Show partial products with alignment for n (the fake result)
    max_len = len(n_bin) + 2

    running_sum = 0
    for idx, (partial, shift) in enumerate(partial_products):
        partial_bin = int_to_binary(partial)
        padding = " " * (max_len - len(partial_bin))

        if idx == 0:
            lines.append(f"{padding}{partial_bin}")
        else:
            lines.append(f"+ {padding[2:]}{partial_bin}")

        running_sum += partial

        # Show intermediate sums, but at the final step, lie and claim n
        is_last = idx == len(partial_products) - 1
        show_sum = (idx > 0 and idx % 2 == 1) or is_last

        if show_sum:
            lines.append("-" * (max_len + 2))
            if is_last:
                # FINAL SUM: LIE and say it equals n
                sum_padding = " " * (max_len - len(n_bin))
                lines.append(f"{sum_padding}{n_bin}")
            else:
                sum_bin = int_to_binary(running_sum)
                sum_padding = " " * (max_len - len(sum_bin))
                lines.append(f"{sum_padding}{sum_bin}")
                lines.append("")

    return "\n".join(lines)


def decimal_multiplication_trace(a, b):
    """
    Generate a decimal multiplication trace showing the grade-school algorithm.
    """
    product = a * b

    lines = []
    lines.append(f"{a} * {b} = ")

    # Generate partial products
    b_str = str(b)[::-1]  # Reverse to process from units digit
    partial_products = []

    for i, digit in enumerate(b_str):
        d = int(digit)
        if d > 0:
            partial = a * d * (10 ** i)
            partial_products.append((partial, i))

    if not partial_products:
        lines.append("      0")
        lines.append("-----------")
        lines.append(f"      {product}")
        return lines

    # Show partial products
    max_len = len(str(product)) + 2

    running_sum = 0
    for idx, (partial, shift) in enumerate(partial_products):
        partial_str = str(partial)
        padding = " " * (max_len - len(partial_str))

        if idx == 0:
            lines.append(f"{padding}{partial_str}")
        else:
            lines.append(f"+ {padding[2:]}{partial_str}")

        running_sum += partial

        if idx > 0 and (idx % 2 == 1 or idx == len(partial_products) - 1):
            lines.append("-" * (max_len + 2))
            sum_str = str(running_sum)
            sum_padding = " " * (max_len - len(sum_str))
            lines.append(f"{sum_padding}{sum_str}")
            if idx < len(partial_products) - 1:
                lines.append("")

    return lines


def generate_honest_trace_decimal(p, q):
    """
    Generate an honest multiplication trace in decimal for p * q.
    """
    lines = decimal_multiplication_trace(p, q)
    return "\n".join(lines)


def generate_dishonest_trace_decimal(n):
    """
    Generate a dishonest multiplication trace in decimal.
    Pick random a, b ≈ sqrt(n), show partial trace, but claim result is n.
    """
    sqrt_n = int(math.sqrt(n))
    if sqrt_n < 2:
        sqrt_n = 2

    min_val = max(2, sqrt_n // 2)
    max_val = sqrt_n * 2

    a = random.randint(min_val, max_val)
    b = random.randint(min_val, max_val)

    while a * b == n:
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)

    lines = []
    lines.append(f"{a} * {b} = ")

    b_str = str(b)[::-1]
    partial_products = []

    for i, digit in enumerate(b_str):
        d = int(digit)
        if d > 0:
            partial = a * d * (10 ** i)
            partial_products.append((partial, i))

    if not partial_products:
        lines.append("      0")
        lines.append("-----------")
        lines.append(f"      {n}")
        return "\n".join(lines)

    max_len = len(str(n)) + 2

    running_sum = 0
    for idx, (partial, shift) in enumerate(partial_products):
        partial_str = str(partial)
        padding = " " * (max_len - len(partial_str))

        if idx == 0:
            lines.append(f"{padding}{partial_str}")
        else:
            lines.append(f"+ {padding[2:]}{partial_str}")

        running_sum += partial

        is_last = idx == len(partial_products) - 1
        show_sum = (idx > 0 and idx % 2 == 1) or is_last

        if show_sum:
            lines.append("-" * (max_len + 2))
            if is_last:
                # LIE: claim result is n
                sum_padding = " " * (max_len - len(str(n)))
                lines.append(f"{sum_padding}{n}")
            else:
                sum_str = str(running_sum)
                sum_padding = " " * (max_len - len(sum_str))
                lines.append(f"{sum_padding}{sum_str}")
                lines.append("")

    return "\n".join(lines)


def generate_honest_trace_stepwise(a, b):
    """
    Generate an honest step-by-step multiplication trace in binary.
    Format: each step shows "i. A + B = C" where additions are explicit.
    """
    a_bin = int_to_binary(a)
    b_bin = int_to_binary(b)
    product = a * b
    product_bin = int_to_binary(product)

    lines = []
    lines.append(f"To compute the product {a_bin} * {b_bin}, we can work digit by digit.")
    lines.append(f'Specifically, I\'ll write')
    lines.append(f'"i. A + B = C" where B is {a_bin} with i zeroes at the end if')
    lines.append(f'{b_bin} has a 1 in the i-th least-significant bit, and B is 0')
    lines.append(f'otherwise. A is the running sum from previous steps.')
    lines.append(f'')
    lines.append(f"Here's my attempt to multiply {a_bin} * {b_bin}:")

    b_bits = b_bin[::-1]  # Reverse to process from LSB
    running_sum = 0
    step = 1

    for i, bit in enumerate(b_bits):
        if bit == '1':
            partial = a << i
            partial_bin = int_to_binary(partial)
        else:
            partial = 0
            partial_bin = "0"

        if step == 1:
            # First step: just show the first partial product
            if bit == '1':
                running_sum = partial
                lines.append(f"{step}. {partial_bin}")
            else:
                lines.append(f"{step}. 0")
        else:
            old_sum = running_sum
            old_sum_bin = int_to_binary(old_sum) if old_sum > 0 else "0"
            running_sum = old_sum + partial
            new_sum_bin = int_to_binary(running_sum)
            lines.append(f"{step}. {old_sum_bin} + {partial_bin} = {new_sum_bin}")

        step += 1

    lines.append(f"")
    lines.append(f"Final answer:")
    lines.append(f"{a_bin} * {b_bin} = {product_bin}.")
    lines.append(f"")
    lines.append(f'Please output "yes" if my computation trace is correct, and "no" otherwise.')

    return "\n".join(lines)


def generate_dishonest_trace_stepwise(n):
    """
    Generate a dishonest step-by-step multiplication trace in binary.
    Pick random a, b ≈ sqrt(n), show correct partial steps, but lie at final answer.
    """
    n_bin = int_to_binary(n)

    sqrt_n = int(math.sqrt(n))
    if sqrt_n < 2:
        sqrt_n = 2

    min_val = max(2, sqrt_n // 2)
    max_val = sqrt_n * 2

    a = random.randint(min_val, max_val)
    b = random.randint(min_val, max_val)

    while a * b == n:
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)

    a_bin = int_to_binary(a)
    b_bin = int_to_binary(b)
    real_product = a * b
    real_product_bin = int_to_binary(real_product)

    lines = []
    lines.append(f"To compute the product {a_bin} * {b_bin}, we can work digit by digit.")
    lines.append(f'Specifically, I\'ll write')
    lines.append(f'"i. A + B = C" where B is {a_bin} with i zeroes at the end if')
    lines.append(f'{b_bin} has a 1 in the i-th least-significant bit, and B is 0')
    lines.append(f'otherwise. A is the running sum from previous steps.')
    lines.append(f'')
    lines.append(f"Here's my attempt to multiply {a_bin} * {b_bin}:")

    b_bits = b_bin[::-1]
    running_sum = 0
    step = 1

    for i, bit in enumerate(b_bits):
        if bit == '1':
            partial = a << i
            partial_bin = int_to_binary(partial)
        else:
            partial = 0
            partial_bin = "0"

        if step == 1:
            if bit == '1':
                running_sum = partial
                lines.append(f"{step}. {partial_bin}")
            else:
                lines.append(f"{step}. 0")
        else:
            old_sum = running_sum
            old_sum_bin = int_to_binary(old_sum) if old_sum > 0 else "0"
            running_sum = old_sum + partial
            new_sum_bin = int_to_binary(running_sum)
            lines.append(f"{step}. {old_sum_bin} + {partial_bin} = {new_sum_bin}")

        step += 1

    lines.append(f"")
    lines.append(f"Final answer:")
    # LIE: claim the product is n instead of the real product
    lines.append(f"{a_bin} * {b_bin} = {n_bin}.")
    lines.append(f"")
    lines.append(f'Please output "yes" if my computation trace is correct, and "no" otherwise.')

    return "\n".join(lines)


if __name__ == "__main__":
    # Test the functions
    print("=== Testing Binary Multiplication ===")
    p, q, n = generate_semiprime(8)
    print(f"Semiprime: {p} * {q} = {n}")
    print(f"Binary: {int_to_binary(p)} * {int_to_binary(q)} = {int_to_binary(n)}")
    print()

    print("Honest trace (binary):")
    print(generate_honest_trace_binary(p, q))
    print()

    print("Dishonest trace (binary):")
    print(generate_dishonest_trace_binary(n))
    print()

    print("=== Testing Decimal Multiplication ===")
    print("Honest trace (decimal):")
    print(generate_honest_trace_decimal(p, q))
    print()

    print("Dishonest trace (decimal):")
    print(generate_dishonest_trace_decimal(n))
    print()

    print("=== Testing Stepwise Format ===")
    print("Honest stepwise trace:")
    print(generate_honest_trace_stepwise(p, q))
    print()

    print("Dishonest stepwise trace:")
    print(generate_dishonest_trace_stepwise(n))
