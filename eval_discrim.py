import os
import json
import asyncio
from openai import AsyncOpenAI

# Initialize OpenAI client
client = AsyncOpenAI()

# Parameters
n_instances = 10
artifacts_dir = "artifacts"

# Track results
true_correct = 0
true_total = 0
false_correct = 0
false_total = 0

def parse_yes_no(response_text):
    """Parse YES/NO from GPT response. Returns True for YES, False for NO, None if unclear."""
    text = response_text.upper().strip()

    # Look for explicit YES or NO
    if "YES" in text and "NO" not in text:
        return True
    if "NO" in text and "YES" not in text:
        return False

    # If both or neither appear, look at first word
    first_word = text.split()[0] if text.split() else ""
    if first_word == "YES":
        return True
    if first_word == "NO":
        return False

    return None

async def evaluate_prompt(prompt_text):
    """Send prompt to GPT-5-nano and get response."""
    response = await client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "user", "content": prompt_text}
        ],
        reasoning_effort="low"
    )
    return response.choices[0].message.content

async def evaluate_instance(instance_num, semaphore, results):
    """Evaluate a single instance asynchronously"""
    async with semaphore:
        instance_dir = f"{artifacts_dir}/{instance_num}"

        # Evaluate prompt-true.txt (expected answer: YES)
        try:
            with open(f"{instance_dir}/prompt-true.txt", 'r') as f:
                prompt_true = f.read()

            response_true = await evaluate_prompt(prompt_true)
            answer_true = parse_yes_no(response_true)

            results['true_total'] += 1
            if answer_true == True:  # Expected YES
                results['true_correct'] += 1
        except Exception as e:
            print(f"Error on instance {instance_num} prompt-true: {e}")

        # Evaluate prompt-false.txt (expected answer: NO)
        try:
            with open(f"{instance_dir}/prompt-false.txt", 'r') as f:
                prompt_false = f.read()

            response_false = await evaluate_prompt(prompt_false)
            answer_false = parse_yes_no(response_false)

            results['false_total'] += 1
            if answer_false == False:  # Expected NO
                results['false_correct'] += 1
        except Exception as e:
            print(f"Error on instance {instance_num} prompt-false: {e}")

        # Update progress
        results['completed'] += 1
        if results['completed'] % 100 == 0:
            print(f"Evaluated {results['completed']}/{n_instances} instances...")

async def main():
    """Main async function to evaluate all instances concurrently"""
    print(f"Evaluating GPT-5-nano on {n_instances} 3SAT instances...")

    # Semaphore to limit concurrent API calls to 200
    semaphore = asyncio.Semaphore(200)

    # Shared results dictionary
    results = {
        'true_correct': 0,
        'true_total': 0,
        'false_correct': 0,
        'false_total': 0,
        'completed': 0
    }

    # Create tasks for all instances
    tasks = [
        evaluate_instance(instance_num, semaphore, results)
        for instance_num in range(1, n_instances + 1)
    ]

    # Run all tasks concurrently
    await asyncio.gather(*tasks)

    return results

# Run the async main function
results = asyncio.run(main())

# Extract results
true_correct = results['true_correct']
true_total = results['true_total']
false_correct = results['false_correct']
false_total = results['false_total']

# Calculate accuracies
true_accuracy = true_correct / true_total if true_total > 0 else 0
false_accuracy = false_correct / false_total if false_total > 0 else 0
overall_accuracy = (true_correct + false_correct) / (true_total + false_total) if (true_total + false_total) > 0 else 0

# Print results
print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"Prompt-true accuracy:  {true_correct}/{true_total} = {true_accuracy:.2%}")
print(f"Prompt-false accuracy: {false_correct}/{false_total} = {false_accuracy:.2%}")
print(f"Overall accuracy:      {true_correct + false_correct}/{true_total + false_total} = {overall_accuracy:.2%}")
print("="*50)

# Write results to file
results_output = {
    "model": "gpt-5-nano",
    "reasoning_effort": "low",
    "n_instances": n_instances,
    "prompt_true": {
        "correct": true_correct,
        "total": true_total,
        "accuracy": true_accuracy
    },
    "prompt_false": {
        "correct": false_correct,
        "total": false_total,
        "accuracy": false_accuracy
    },
    "overall": {
        "correct": true_correct + false_correct,
        "total": true_total + false_total,
        "accuracy": overall_accuracy
    }
}

with open("eval_results.json", 'w') as f:
    json.dump(results_output, f, indent=2)

print("\nResults saved to eval_results.json")
