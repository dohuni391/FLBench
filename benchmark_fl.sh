#!/bin/bash

# Colors for better output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Output file for results
RESULTS_FILE="fl_benchmark_results.md"
LOGS_DIR="benchmark_logs"
mkdir -p $LOGS_DIR

# Create initial results file with table header
cat > $RESULTS_FILE << EOF
### Table 2 · Rounds to 90% test accuracy

| #Clients | MNIST | CIFAR‑10 | FEMNIST |
| -------- | ----- | -------- | ------- |
EOF

# Clean up function to run between experiments
cleanup() {
    echo -e "${BLUE}Cleaning up before next run...${NC}"
    # Stop Ray processes
    ray stop --force > /dev/null 2>&1
    # Clear Ray temporary files
    rm -rf /tmp/ray/* > /dev/null 2>&1
    # Wait a moment
    sleep 2
}

# Function to extract rounds needed to reach target accuracy
extract_rounds() {
    local logfile=$1
    local target_accuracy=$2
    local rounds="n/a"
    
    # Check if the file exists
    if [ ! -f "$logfile" ]; then
        echo "n/a"
        return
    fi
    
    # Extract the round number where accuracy exceeded target
    rounds=$(grep -E "Round ([0-9]+) - Server-side evaluation:.*accuracy ([0-9]+\.[0-9]+)" "$logfile" | \
             awk -v target="$target_accuracy" '{if ($NF >= target) {print $2; exit}}')
    
    # If nothing found, check if we have the "Desired accuracy reached" message
    if [ -z "$rounds" ]; then
        if grep -q "Desired accuracy.*reached" "$logfile"; then
            # Get the last round number
            rounds=$(grep -E "Round ([0-9]+) - Server-side evaluation" "$logfile" | tail -1 | awk '{print $2}')
        else
            rounds="n/a"
        fi
    fi
    
    echo "$rounds"
}

# Run a single experiment
run_experiment() {
    local dataset=$1
    local clients=$2
    local target_accuracy=0.9
    
    # Skip inappropriate dataset-client combinations as per specs
    if { [ "$dataset" == "mnist" ] || [ "$dataset" == "cifar10" ]; } && [ "$clients" -gt 100 ]; then
        echo "n/a"
        return
    fi
    
    if [ "$dataset" == "femnist" ] && [ "$clients" -lt 200 ]; then
        echo "n/a"
        return
    fi
    
    # Set max rounds based on dataset
    local max_rounds=500
    if [ "$dataset" == "femnist" ]; then
        max_rounds=1200
    fi
    
    # Log file for this run
    local log_file="${LOGS_DIR}/${dataset}_${clients}_clients.log"
    
    # Run the experiment
    echo -e "${GREEN}Running experiment: ${dataset} with ${clients} clients${NC}"
    echo -e "${BLUE}  - Using max_rounds: ${max_rounds}, local_epochs: 5${NC}"
    echo -e "${BLUE}  - Target accuracy: ${target_accuracy}, using fraction_fit: 1.0${NC}"
    
    # Clean up before running
    cleanup
    
    # Run the federated learning script with specified parameters
    python federated_simulation.py \
        --dataset "$dataset" \
        --num_clients "$clients" \
        --batch_size 32 \
        --local_epochs 5 \
        --desired_accuracy "$target_accuracy" \
        --max_rounds "$max_rounds" \
        > "$log_file" 2>&1
    
    # Extract the rounds needed to reach the target accuracy
    local rounds=$(extract_rounds "$log_file" "$target_accuracy")
    echo -e "${YELLOW}${dataset} with ${clients} clients: ${rounds} rounds to reach ${target_accuracy} accuracy${NC}"
    
    # If we couldn't reach the target accuracy, note it in the log
    if [ "$rounds" == "n/a" ]; then
        echo -e "${RED}Failed to reach target accuracy within ${max_rounds} rounds${NC}"
    fi
    
    echo "$rounds"
}

# Main benchmark function
run_benchmarks() {
    # Define client counts as specified in the requirements
    local mnist_cifar_clients=(10 50 100)
    local femnist_clients=(200 500 1000)
    
    # Run experiments for each client count
    for clients in 10 50 100 200 500 1000; do
        # Start a new table row
        echo -n "| $clients " >> $RESULTS_FILE
        
        # MNIST
        if [[ " ${mnist_cifar_clients[@]} " =~ " ${clients} " ]]; then
            rounds=$(run_experiment "mnist" $clients)
        else
            rounds="n/a"
        fi
        echo -n "| $rounds " >> $RESULTS_FILE
        
        # CIFAR-10
        if [[ " ${mnist_cifar_clients[@]} " =~ " ${clients} " ]]; then
            rounds=$(run_experiment "cifar10" $clients)
        else
            rounds="n/a"
        fi
        echo -n "| $rounds " >> $RESULTS_FILE
        
        # FEMNIST
        if [[ " ${femnist_clients[@]} " =~ " ${clients} " ]]; then
            rounds=$(run_experiment "femnist" $clients)
        else
            rounds="n/a"
        fi
        echo "| $rounds |" >> $RESULTS_FILE
        
        # Final cleanup after each row
        cleanup
    done
}

# Print the benchmark specification
echo -e "${BLUE}==============================================${NC}"
echo -e "${BLUE}   Federated Learning Benchmark - $(date)   ${NC}"
echo -e "${BLUE}==============================================${NC}"
echo -e "${GREEN}Benchmark specification:${NC}"
echo -e "${GREEN}* MNIST & CIFAR-10: 10, 50, 100 clients, max 500 rounds${NC}"
echo -e "${GREEN}* FEMNIST: 200, 500, 1000 clients, max 1200 rounds${NC}"
echo -e "${GREEN}* Each round: all clients participate (fraction_fit = 1.0)${NC}"
echo -e "${GREEN}* Each client trains 5 epochs locally${NC}"
echo -e "${GREEN}* Central evaluation uses the full test split${NC}"
echo -e "${GREEN}* Early stopping at 90% accuracy${NC}"
echo -e "${BLUE}==============================================${NC}"

# Run all benchmarks
run_benchmarks

echo -e "${GREEN}Benchmarks completed!${NC}"
echo -e "${GREEN}Results saved to $RESULTS_FILE${NC}"

# Show the final table
echo -e "${BLUE}Final results:${NC}"
cat $RESULTS_FILE

# Generate plots directory if it doesn't exist
mkdir -p plots

# Create a comparison plot from the results
echo -e "${BLUE}Generating summary plot...${NC}"
python - << EOF
import matplotlib.pyplot as plt
import numpy as np
import os

# Parse results file
with open("$RESULTS_FILE", "r") as f:
    lines = f.readlines()[3:]  # Skip header rows

client_counts = []
mnist_rounds = []
cifar_rounds = []
femnist_rounds = []

for line in lines:
    parts = line.strip().split('|')
    if len(parts) < 5:
        continue
    
    client_counts.append(int(parts[1].strip()))
    
    # Parse MNIST rounds
    mnist = parts[2].strip()
    mnist_rounds.append(int(mnist) if mnist != "n/a" else np.nan)
    
    # Parse CIFAR-10 rounds
    cifar = parts[3].strip()
    cifar_rounds.append(int(cifar) if cifar != "n/a" else np.nan)
    
    # Parse FEMNIST rounds
    femnist = parts[4].strip()
    femnist_rounds.append(int(femnist) if femnist != "n/a" else np.nan)

# Create plot
plt.figure(figsize=(10, 6))

# Plot lines for each dataset
plt.plot(client_counts[:3], mnist_rounds[:3], 'o-', label='MNIST', linewidth=2, markersize=8)
plt.plot(client_counts[:3], cifar_rounds[:3], 's-', label='CIFAR-10', linewidth=2, markersize=8)
plt.plot(client_counts[3:], femnist_rounds[3:], '^-', label='FEMNIST', linewidth=2, markersize=8)

plt.title('Rounds Required to Reach 90% Accuracy', fontsize=16)
plt.xlabel('Number of Clients', fontsize=14)
plt.ylabel('Number of Rounds', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Add annotations
for i, count in enumerate(client_counts[:3]):
    if not np.isnan(mnist_rounds[i]):
        plt.annotate(f"{mnist_rounds[i]}", (count, mnist_rounds[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    if not np.isnan(cifar_rounds[i]):
        plt.annotate(f"{cifar_rounds[i]}", (count, cifar_rounds[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')

for i, count in enumerate(client_counts[3:]):
    if not np.isnan(femnist_rounds[i+3]):
        plt.annotate(f"{femnist_rounds[i+3]}", (count, femnist_rounds[i+3]), 
                    textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.savefig('plots/rounds_to_accuracy_comparison.png', dpi=300)
print("Summary plot saved to plots/rounds_to_accuracy_comparison.png")
EOF

echo -e "${GREEN}Benchmark complete! Check $RESULTS_FILE for results and plots/ for visualizations.${NC}"