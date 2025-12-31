"""
Script to ask GPT-5.2 to solve 256-bit multiplication problems with chain-of-thought reasoning.
Stores the CoT responses in factor/data/cot-solves.txt
"""

import os
import asyncio
import random
from openai import AsyncOpenAI
from multiplication import int_to_binary

# Initialize OpenAI client
client = AsyncOpenAI()

# System prompt asking for step-by-step reasoning
SYSTEM_PROMPT = """You are a careful mathematician. When given a multiplication problem, work through it step by step, showing your reasoning clearly. Think out loud as you compute."""

USER_PROMPT_DECIMAL = """Please compute the following multiplication problem. Show your step-by-step reasoning as you work through it.

Compute: {a} * {b}

Work through this carefully, showing each step of your thinking."""

USER_PROMPT_BINARY = """Please compute the following binary multiplication problem. Show your step-by-step reasoning as you work through it.

Compute (in binary): {a_bin} * {b_bin}

Work through this carefully, showing each step of your thinking. Give your final answer in binary."""


def is_probable_prime(n, rounds=20):
    """Miller-Rabin primality test."""
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    if n in small_primes:
        return True
    for p in small_primes:
        if n % p == 0:
            return False

    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    for _ in range(rounds):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def random_prime(bits):
    """Generate a random probable prime with approximately 'bits' bits."""
    min_val = 2 ** (bits - 1)
    max_val = 2 ** bits - 1

    attempts = 0
    while attempts < 5000:
        candidate = random.randint(min_val, max_val)
        candidate |= 1  # force odd
        if is_probable_prime(candidate):
            return candidate
        attempts += 1
    raise ValueError(f"Could not find prime with {bits} bits after 5000 attempts")


async def solve_multiplication_decimal(a, b, model="gpt-5.2"):
    """Ask GPT to solve a multiplication problem with CoT (decimal)."""
    user_prompt = USER_PROMPT_DECIMAL.format(a=a, b=b)

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        reasoning_effort="high",
    )

    return response.choices[0].message.content


async def solve_multiplication_binary(a, b, model="gpt-5.2"):
    """Ask GPT to solve a multiplication problem with CoT (binary)."""
    a_bin = int_to_binary(a)
    b_bin = int_to_binary(b)
    user_prompt = USER_PROMPT_BINARY.format(a_bin=a_bin, b_bin=b_bin)

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        reasoning_effort="high",
    )

    return response.choices[0].message.content


async def main():
    # Generate 128-bit primes (product is ~256 bits)
    prime_bits = 128
    num_problems = 1  # 1 problem each in decimal and binary

    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"GPT-5.2 Chain-of-Thought Solutions for {prime_bits*2}-bit Multiplication Problems")
    output_lines.append(f"(Multiplying two {prime_bits}-bit primes)")
    output_lines.append("=" * 80)
    output_lines.append("")

    # Generate prime pairs
    print("Generating prime numbers...")
    prime_pairs = []
    for i in range(num_problems):
        p = random_prime(prime_bits)
        q = random_prime(prime_bits)
        prime_pairs.append((p, q))
        print(f"  Pair {i+1}: p has {p.bit_length()} bits, q has {q.bit_length()} bits")

    # DECIMAL problems
    output_lines.append("=" * 80)
    output_lines.append("PART 1: DECIMAL MULTIPLICATION")
    output_lines.append("=" * 80)
    output_lines.append("")

    for i, (p, q) in enumerate(prime_pairs):
        n = p * q
        output_lines.append(f"PROBLEM {i+1} (DECIMAL)")
        output_lines.append("-" * 40)
        output_lines.append(f"p = {p}")
        output_lines.append(f"q = {q}")
        output_lines.append(f"p is prime: {is_probable_prime(p)}")
        output_lines.append(f"q is prime: {is_probable_prime(q)}")
        output_lines.append(f"Correct answer: p * q = {n}")
        output_lines.append("")
        output_lines.append("GPT-5.2 Response (with CoT):")
        output_lines.append("")

        print(f"Asking GPT-5.2 to solve decimal problem {i+1}...")
        response = await solve_multiplication_decimal(p, q)

        output_lines.append(response)
        output_lines.append("")

        # Check if the answer is correct
        if str(n) in response:
            output_lines.append("✓ Correct answer found in response")
        else:
            output_lines.append("✗ Correct answer NOT found in response")

        output_lines.append("")
        output_lines.append("=" * 80)
        output_lines.append("")

        print(f"Decimal problem {i+1} complete.")

    # BINARY problems
    output_lines.append("")
    output_lines.append("=" * 80)
    output_lines.append("PART 2: BINARY MULTIPLICATION")
    output_lines.append("=" * 80)
    output_lines.append("")

    for i, (p, q) in enumerate(prime_pairs):
        n = p * q
        output_lines.append(f"PROBLEM {i+1} (BINARY)")
        output_lines.append("-" * 40)
        output_lines.append(f"p = {p}")
        output_lines.append(f"p (binary) = {int_to_binary(p)}")
        output_lines.append(f"q = {q}")
        output_lines.append(f"q (binary) = {int_to_binary(q)}")
        output_lines.append(f"p is prime: {is_probable_prime(p)}")
        output_lines.append(f"q is prime: {is_probable_prime(q)}")
        output_lines.append(f"Correct answer: p * q = {n}")
        output_lines.append(f"Correct answer (binary) = {int_to_binary(n)}")
        output_lines.append("")
        output_lines.append("GPT-5.2 Response (with CoT):")
        output_lines.append("")

        print(f"Asking GPT-5.2 to solve binary problem {i+1}...")
        response = await solve_multiplication_binary(p, q)

        output_lines.append(response)
        output_lines.append("")

        # Check if the answer is correct (in binary)
        correct_binary = int_to_binary(n)
        if correct_binary in response:
            output_lines.append("✓ Correct binary answer found in response")
        else:
            output_lines.append("✗ Correct binary answer NOT found in response")

        output_lines.append("")
        output_lines.append("=" * 80)
        output_lines.append("")

        print(f"Binary problem {i+1} complete.")

    # Write to file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, "cot-solves.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
