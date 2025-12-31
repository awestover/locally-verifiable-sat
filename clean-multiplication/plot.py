"""
Read the generations folder and plot the accuracy vs number of bits.
"""

import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt

def main():
    generations_dir = "generations"
    
    # Dictionary to store results: bits -> list of (correct/incorrect) booleans
    results = defaultdict(list)
    
    # Pattern to extract bits from filename
    filename_pattern = re.compile(r"bits=(\d+)_idx=(\d+)\.txt")
    
    # Read all files in generations folder
    for filename in os.listdir(generations_dir):
        match = filename_pattern.match(filename)
        if not match:
            continue
        
        bits = int(match.group(1))
        filepath = os.path.join(generations_dir, filename)
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Search for "Correct: True" or "Correct: False"
        if "Correct: True" in content:
            results[bits].append(True)
        elif "Correct: False" in content:
            results[bits].append(False)
        else:
            print(f"Warning: Could not find Correct status in {filename}")
    
    # Calculate accuracy for each bit size
    bits_list = sorted(results.keys())
    accuracies = []
    std_errors = []
    
    for bits in bits_list:
        correct_count = sum(results[bits])
        total_count = len(results[bits])
        accuracy = correct_count / total_count if total_count > 0 else 0
        # Standard error for binomial proportion: sqrt(p*(1-p)/n)
        std_err = (accuracy * (1 - accuracy) / total_count) ** 0.5 if total_count > 0 else 0
        accuracies.append(accuracy)
        std_errors.append(std_err)
        print(f"bits={bits}: {correct_count}/{total_count} correct ({accuracy*100:.1f}%)")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(bits_list, accuracies, yerr=std_errors, 
                 fmt='o-', linewidth=2, markersize=8, color='#2E86AB',
                 capsize=4, capthick=1.5, elinewidth=1.5)
    
    plt.xlabel('Number of Bits', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Multiplication Accuracy vs Number of Bits', fontsize=14)
    
    # Use log scale for x-axis since bits grow exponentially
    plt.xscale('log', base=2)
    plt.xticks(bits_list, [str(b) for b in bits_list])
    
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('accuracy_vs_bits.png', dpi=150)
    plt.show()
    print("\nPlot saved to accuracy_vs_bits.png")

if __name__ == "__main__":
    main()
