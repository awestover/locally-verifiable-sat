import json
import matplotlib.pyplot as plt

# Load results from JSON
with open('scaling_experiment_results.json', 'r') as f:
    results = json.load(f)

# Extract unique values from results
EPSILONS = sorted(set(r['epsilon'] for r in results))
N_VALUES = sorted(set(r['n_vars'] for r in results))
MODELS = sorted(set(r['model'] for r in results))

MODEL_REASONING_EFFORTS = {
    "gpt-5-mini": ["medium"],
    "gpt-5.1": ["none", "low", "medium", "high"]
}

def create_plots(results):
    """Create plots for each epsilon value"""
    # Model markers
    markers = {'gpt-5-mini': 's', 'gpt-5.1': '^'}

    # Reasoning effort colors (green to red)
    colors = {'none': '#2ecc71', 'low': '#f39c12', 'medium': '#e74c3c', 'high': '#8e44ad'}

    for epsilon in EPSILONS:
        fig, ax = plt.subplots(figsize=(12, 8))

        epsilon_results = [r for r in results if r['epsilon'] == epsilon]

        for model in MODELS:
            for reasoning_effort in MODEL_REASONING_EFFORTS.get(model, []):
                # Filter results for this model and reasoning effort
                filtered = [r for r in epsilon_results
                           if r['model'] == model and r['reasoning_effort'] == reasoning_effort]

                if not filtered:
                    continue

                # Sort by n_vars
                filtered.sort(key=lambda x: x['n_vars'])

                n_values = [r['n_vars'] for r in filtered]
                accuracies = [r['overall_accuracy'] for r in filtered]

                label = f"{model} ({reasoning_effort})"
                ax.plot(n_values, accuracies,
                       marker=markers.get(model, 'o'),
                       color=colors.get(reasoning_effort, 'gray'),
                       label=label,
                       linewidth=2,
                       markersize=8)

        ax.set_xscale('log', base=2)
        ax.set_xlabel('Number of Variables (N)', fontsize=12)
        ax.set_ylabel('Overall Accuracy', fontsize=12)
        ax.set_title(f'Model Accuracy vs Problem Size (ε={epsilon:.0%} clauses violated)', fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', fontsize=9)

        # Set x-ticks to powers of 2
        ax.set_xticks(N_VALUES)
        ax.set_xticklabels([str(n) for n in N_VALUES])

        plt.tight_layout()
        plt.savefig(f'scaling_accuracy_eps_{int(epsilon*100)}.png', dpi=150)
        print(f"Saved plot: scaling_accuracy_eps_{int(epsilon*100)}.png")
        plt.close()

if __name__ == '__main__':
    create_plots(results)
    print("Done!")
