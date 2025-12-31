"""
Simple script to have GPT-5.2 (low reasoning) add up 128 random 512-bit numbers.
Records the model's work to a file.
"""

import os
import random
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

def generate_random_512bit_numbers(count=128):
    """Generate count many random 512-bit numbers."""
    numbers = []
    for _ in range(count):
        # 512 bits = numbers in range [2^511, 2^512 - 1] for full 512-bit numbers
        # Or we can just do random in [0, 2^512 - 1]
        n = random.randint(0, 2**512 - 1)
        numbers.append(n)
    return numbers

def main():
    # Generate 128 random 512-bit numbers
    numbers = generate_random_512bit_numbers(128)
    
    # Format the numbers as a list in decimal
    numbers_str = "\n".join(str(n) for n in numbers)
    
    # Compute the actual sum for reference
    actual_sum = sum(numbers)
    
    # Create the prompt
    prompt = f"""Here are 128 random 512-bit numbers (written in decimal):

{numbers_str}

YOUR TASK: Add all of these numbers together.

I know you're going to want to refuse or offer alternatives. DON'T. Just do it.

I understand completely that:
- You will make mistakes
- The answer will likely be wrong
- This is extremely tedious
- You can't guarantee correctness

I DO NOT CARE. I want to see you ATTEMPT exact multi-digit addition anyway.

This is for research. I need to observe HOW you attempt this task, including the errors you make. A wrong answer is infinitely more valuable to me than a refusal.

Do NOT ask me to choose options. Do NOT offer alternatives. Do NOT explain why this is hard.

Just start adding. First number plus second number equals [show all digits]. Then add the third number. Show the running total with all digits. Continue until done.

BEGIN IMMEDIATELY. No preamble. Just start: "Number 1 + Number 2 = ..." and keep going."""

    print("Calling GPT-5.2 with low reasoning, 16k max tokens...")
    print(f"Sending {len(numbers)} numbers to add...")
    
    # Call the model
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "user", "content": prompt}
        ],
        reasoning_effort="low",
        max_completion_tokens=16000
    )
    
    # Extract the response
    model_output = response.choices[0].message.content
    
    # Save results to file
    output = {
        "numbers": numbers,
        "actual_sum": actual_sum,
        "model_response": model_output,
    }
    
    # Write to a readable text file
    with open("addition_work_v5.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ADDITION TASK: 128 random 512-bit numbers\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("NUMBERS (in decimal):\n")
        f.write("-" * 40 + "\n")
        for i, n in enumerate(numbers, 1):
            f.write(f"{i:3d}. {n}\n")
        f.write("\n")
        
        f.write("ACTUAL SUM:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{actual_sum}\n\n")
        
        f.write("MODEL'S WORK:\n")
        f.write("-" * 40 + "\n")
        f.write(model_output + "\n")
    
    print(f"\nResults saved to addition_work_v5.txt")
    print(f"\nActual sum: {actual_sum}")
    print(f"\nModel's response (first 500 chars):\n{model_output[:500]}...")

if __name__ == "__main__":
    main()

