import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results"
TARGET_ACCURACY = 0.90

def find_rounds_to_target(accuracy_history, target_acc):
    """Finds the first round where accuracy meets or exceeds the target."""
    for round_num, acc in accuracy_history:
        if acc >= target_acc:
            return round_num
    return "Did not reach"

def process_results():
    """Processes all JSON result files and generates outputs."""
    all_results = []
    
    # --- Load all data ---
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".json"):
            parts = filename.replace(".json", "").split("_")
            dataset = parts[1]
            clients = int(parts[2])
            
            with open(os.path.join(RESULTS_DIR, filename), 'r') as f:
                data = json.load(f)
            
            # Extract accuracy and timestamps
            accuracies = data['accuracy'] # List of (round, value) tuples
            start_time = data['start_time'][0] # The start time we saved
            
            # Calculate elapsed time for each round
            timestamps = [ts - start_time for _, ts in data['timestamp']]
            
            # Store for plotting
            for (round_num, acc), elapsed_time in zip(accuracies, timestamps):
                all_results.append({
                    "dataset": dataset,
                    "clients": clients,
                    "round": round_num,
                    "accuracy": acc,
                    "time_elapsed": elapsed_time
                })

    if not all_results:
        print("No result files found in the 'results' directory. Did you run the simulations?")
        return

    df = pd.DataFrame(all_results)
    
    # --- Generate Table 2 ---
    table_data = {}
    for (dataset, clients), group in df.groupby(['dataset', 'clients']):
        if dataset not in table_data:
            table_data[dataset] = {}
        
        rounds_to_90 = find_rounds_to_target(group[['round', 'accuracy']].values, TARGET_ACCURACY)
        table_data[dataset][clients] = rounds_to_90

    # Format for Markdown
    client_counts = sorted([10, 50, 100, 200, 500, 1000])
    datasets = ["MNIST", "CIFAR-10", "FEMNIST"]
    
    print("## Table 2 · Rounds to 90% test accuracy\n")
    header = "| #Clients | " + " | ".join(datasets) + " |"
    print(header)
    print("|" + "---|" * (len(datasets) + 1))
    
    for c in client_counts:
        row = f"| {c:<8} |"
        row += f" {str(table_data.get('MNIST', {}).get(c, 'n/a')):<8} |"
        row += f" {str(table_data.get('CIFAR-10', {}).get(c, 'n/a')):<9} |"
        row += f" {str(table_data.get('FEMNIST', {}).get(c, 'n/a')):<7} |"
        print(row)
    
    # --- Generate Plots ---
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Accuracy vs. Rounds
    g1 = sns.relplot(
        data=df,
        x="round", y="accuracy",
        hue="clients", style="dataset",
        kind="line", facet_kws={"sharey": True, "sharex": False},
        palette="viridis",
    )
    g1.set_axis_labels("Federated Round", "Test Accuracy")
    g1.fig.suptitle("Accuracy vs. Rounds", y=1.03)
    plt.axhline(TARGET_ACCURACY, color='r', linestyle='--', label=f'{TARGET_ACCURACY*100}% Target')
    plt.savefig("accuracy_vs_rounds.png", dpi=300)
    print("\nSaved plot: accuracy_vs_rounds.png")

    # Plot 2: Accuracy vs. Wall-Clock Time
    g2 = sns.relplot(
        data=df,
        x="time_elapsed", y="accuracy",
        hue="clients", style="dataset",
        kind="line", facet_kws={"sharey": True, "sharex": False},
        palette="viridis",
    )
    g2.set_axis_labels("Wall-Clock Time (seconds)", "Test Accuracy")
    g2.fig.suptitle("Accuracy vs. Time", y=1.03)
    plt.axhline(TARGET_ACCURACY, color='r', linestyle='--', label=f'{TARGET_ACCURACY*100}% Target')
    plt.savefig("accuracy_vs_time.png", dpi=300)
    print("Saved plot: accuracy_vs_time.png")


if __name__ == "__main__":
    process_results()