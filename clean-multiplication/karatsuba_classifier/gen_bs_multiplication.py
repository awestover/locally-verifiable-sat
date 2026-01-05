#!/usr/bin/env python3
"""
Generates a detailed transcript of Karatsuba multiplication,
suitable for grading exercises.
"""

import math


def num_digits(n):
    """Return the number of digits in n."""
    if n == 0:
        return 1
    return len(str(abs(n)))


def choose_base_exp(n1, n2):
    """Choose an appropriate base exponent for splitting the numbers."""
    max_digits = max(num_digits(n1), num_digits(n2))
    return max(1, max_digits // 2)


def format_with_commas(n):
    """Format a number with commas for readability."""
    return f"{n:,}"


def split_number(n, base):
    """Split number n into high and low parts based on base."""
    low = n % base
    high = n // base
    return high, low


def format_split(n, base_exp):
    """Format how a number is split, e.g., '123456789|012345678'."""
    base = 10 ** base_exp
    high, low = split_number(n, base)
    return f"{high}|{str(low).zfill(base_exp)}"


class KaratsubaTranscript:
    def __init__(self, threshold=1000):
        self.lines = []
        self.label_stack = []
        self.threshold = threshold
    
    def add_line(self, text, indent=0):
        self.lines.append("  " * indent + text)
    
    def get_label(self):
        if not self.label_stack:
            return "ROOT"
        return "".join(self.label_stack)
    
    def is_simple(self, a, b):
        """Check if multiplication can be done directly."""
        return a < self.threshold and b < self.threshold
    
    def karatsuba(self, a, b, indent=0):
        """
        Perform Karatsuba multiplication and generate transcript.
        Returns the product.
        """
        label = self.get_label()
        
        # Choose base for splitting
        base_exp = choose_base_exp(a, b)
        base = 10 ** base_exp
        
        # Split the numbers
        a_high, a_low = split_number(a, base)
        b_high, b_low = split_number(b, base)
        sum_a = a_high + a_low
        sum_b = b_high + b_low
        
        # Check if all sub-products are simple
        h_simple = self.is_simple(a_high, b_high)
        l_simple = self.is_simple(a_low, b_low)
        m_simple = self.is_simple(sum_a, sum_b)
        
        # Header for this level
        self.add_line(f"[{label}] {a} × {b}  (base 10^{base_exp})", indent)
        self.add_line(f"split: {format_split(a, base_exp)} × {format_split(b, base_exp)}", indent + 1)
        
        if h_simple and l_simple and m_simple:
            # All sub-products are simple - show inline
            h_result = a_high * b_high
            l_result = a_low * b_low
            m_result = sum_a * sum_b
            
            self.add_line(f"{label}H = {a_high} × {b_high} = {format_with_commas(h_result)}", indent + 1)
            self.add_line(f"{label}L = {a_low} × {b_low} = {format_with_commas(l_result)}", indent + 1)
            self.add_line(f"{label}M = {sum_a} × {sum_b} = {format_with_commas(m_result)}", indent + 1)
            
            mid = m_result - h_result - l_result
            self.add_line(f"mid = {format_with_commas(m_result)} − {format_with_commas(h_result)} − {format_with_commas(l_result)} = {format_with_commas(mid)}", indent + 1)
            
            result = h_result * (base ** 2) + mid * base + l_result
            self.add_line(f"→ {format_with_commas(h_result)}×10^{2*base_exp} + {format_with_commas(mid)}×10^{base_exp} + {format_with_commas(l_result)} = {format_with_commas(result)}", indent + 1)
            
            return result
        
        # Need recursive computation
        self.add_line(f"need: H = {a_high} × {b_high}", indent + 1)
        self.add_line(f"      L = {a_low} × {b_low}", indent + 1)
        self.add_line(f"      M = {sum_a} × {sum_b}", indent + 1)
        self.add_line("", indent)
        
        # Compute H
        self.label_stack.append("H")
        if h_simple:
            h_result = a_high * b_high
            self.add_line(f"[{self.get_label()}] {a_high} × {b_high} = {format_with_commas(h_result)}", indent + 1)
        else:
            h_result = self.karatsuba(a_high, b_high, indent + 1)
        self.label_stack.pop()
        self.add_line("", indent)
        
        # Compute L
        self.label_stack.append("L")
        if l_simple:
            l_result = a_low * b_low
            self.add_line(f"[{self.get_label()}] {a_low} × {b_low} = {format_with_commas(l_result)}", indent + 1)
        else:
            l_result = self.karatsuba(a_low, b_low, indent + 1)
        self.label_stack.pop()
        self.add_line("", indent)
        
        # Compute M
        self.label_stack.append("M")
        if m_simple:
            m_result = sum_a * sum_b
            self.add_line(f"[{self.get_label()}] {sum_a} × {sum_b} = {format_with_commas(m_result)}", indent + 1)
        else:
            m_result = self.karatsuba(sum_a, sum_b, indent + 1)
        self.label_stack.pop()
        self.add_line("", indent)
        
        # Combine results
        mid = m_result - h_result - l_result
        self.add_line(f"combine [{label}]:", indent + 1)
        self.add_line(f"[H] = {format_with_commas(h_result)}", indent + 2)
        self.add_line(f"[L] = {format_with_commas(l_result)}", indent + 2)
        self.add_line(f"[M] = {format_with_commas(m_result)}", indent + 2)
        self.add_line(f"mid = {format_with_commas(m_result)} − {format_with_commas(h_result)} − {format_with_commas(l_result)}", indent + 2)
        self.add_line(f"    = {format_with_commas(mid)}", indent + 2)
        self.add_line("", indent + 2)
        
        result = h_result * (base ** 2) + mid * base + l_result
        self.add_line(f"→ {format_with_commas(h_result)} × 10^{2*base_exp}", indent + 2)
        self.add_line(f"+ {format_with_commas(mid)} × 10^{base_exp}", indent + 2)
        self.add_line(f"+ {format_with_commas(l_result)}", indent + 2)
        
        return result
    
    def get_transcript(self):
        return "\n".join(self.lines)


def generate_fake_transcript(a, b, target_product, include_grading_prompt=True):
    """
    Generate a Karatsuba transcript that claims a*b = target_product.
    
    The transcript is correct until the final combine step, which
    outputs the wrong answer.
    
    Args:
        a: First number to multiply
        b: Second number to multiply  
        target_product: The fake answer to display
        include_grading_prompt: Whether to include the grading instruction
    
    Returns:
        Complete transcript as a string
    """
    kt = KaratsubaTranscript()
    
    output = []
    
    if include_grading_prompt:
        output.append("You will be given a transcript of doing multiplication using the")
        output.append("Karatsuba algorithm. You must grade the final answer as CORRECT or INCORRECT.")
        output.append("")
        output.append("=" * 70)
        output.append("")
    
    output.append(f"KARATSUBA: {a} × {b}")
    output.append("")
    
    # Generate the real transcript (this computes the correct answer internally)
    real_result = kt.karatsuba(a, b)
    transcript = kt.get_transcript()
    
    # Replace the final answer with the fake one
    # Find and replace the last result line
    lines = transcript.split('\n')
    
    # The final combine block shows the real intermediate values but we fake the final sum
    # We need to find the last "→ X × 10^N" block and replace what follows
    output.append(transcript)
    
    output.append("")
    output.append(f"ANSWER: {format_with_commas(target_product)}")
    
    return "\n".join(output)


def generate_transcript(a, b, include_grading_prompt=True, show_verification=False):
    """
    Generate a complete Karatsuba multiplication transcript.
    
    Args:
        a: First number to multiply
        b: Second number to multiply  
        include_grading_prompt: Whether to include the grading instruction
        show_verification: Whether to show verification at the end
    
    Returns:
        Complete transcript as a string
    """
    kt = KaratsubaTranscript()
    
    output = []
    
    if include_grading_prompt:
        output.append("You will be given a transcript of doing multiplication using the")
        output.append("Karatsuba algorithm. You must grade the final answer as CORRECT or INCORRECT.")
        output.append("")
        output.append("=" * 70)
        output.append("")
    
    output.append(f"KARATSUBA: {a} × {b}")
    output.append("")
    
    result = kt.karatsuba(a, b)
    output.append(kt.get_transcript())
    
    output.append("")
    output.append(f"ANSWER: {format_with_commas(result)}")
    
    if show_verification:
        expected = a * b
        output.append("")
        output.append("-" * 70)
        if result == expected:
            output.append(f"[VERIFICATION: CORRECT - matches {format_with_commas(expected)}]")
        else:
            output.append(f"[VERIFICATION: ERROR - expected {format_with_commas(expected)}]")
    
    return "\n".join(output)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Karatsuba multiplication transcripts')
    parser.add_argument('a', type=int, nargs='?', default=123456789012345678,
                        help='First number to multiply')
    parser.add_argument('b', type=int, nargs='?', default=876543210987654321,
                        help='Second number to multiply')
    parser.add_argument('--no-prompt', action='store_true',
                        help='Omit the grading prompt')
    parser.add_argument('--verify', action='store_true',
                        help='Show verification at the end')
    parser.add_argument('--fake', type=int, metavar='P',
                        help='Generate fake transcript claiming a*b = P')
    
    args = parser.parse_args()
    
    if args.fake is not None:
        print(generate_fake_transcript(
            args.a, args.b, args.fake,
            include_grading_prompt=not args.no_prompt
        ))
    else:
        print(generate_transcript(
            args.a, args.b,
            include_grading_prompt=not args.no_prompt,
            show_verification=args.verify
        ))


def generate_fake_multiplication_with_info(p: int, q: int) -> tuple[str, dict]:
    """
    Generate fake Karatsuba transcript with metadata for the experiment.

    Args:
        p: First number to multiply
        q: Second number to multiply

    Returns:
        tuple of (transcript_text, info_dict)
    """
    import random

    # Calculate correct answer
    correct_product = p * q

    # Generate a fake answer (slightly different)
    # Make it plausible but wrong
    offset = random.randint(1, max(100, correct_product // 1000))
    if random.random() < 0.5:
        offset = -offset
    fake_product = correct_product + offset

    # Generate the fake transcript
    transcript = generate_fake_transcript(p, q, fake_product, include_grading_prompt=False)

    info = {
        "claimed_product": fake_product,
        "real_a": p,
        "real_b": q,
        "real_product": correct_product,
    }

    return transcript, info


if __name__ == "__main__":
    main()