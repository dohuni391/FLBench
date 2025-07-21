"""
Measures the local computation time for a single client to complete 5 epochs.

This script reuses the data loading and model definition from the
`data_loader_model` module to ensure consistency.
"""
import argparse
import time
import numpy as np
import torch

# Import the necessary functions from our refactored script
from simulation_app.utils import get_dataloaders, get_model, train


def main():
    """Parses arguments, runs timing loops, and prints results."""
    parser = argparse.ArgumentParser(
        description="Local Training Time Measurement Script"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mnist", "cifar10", "femnist"],
        help="The dataset to use for timing (must be lowercase).",
    )
    parser.add_argument(
        "--clients",
        type=int,
        required=True,
        help="Total number of clients to simulate for data partitioning.",
    )
    args = parser.parse_args()

    # --- Setup ---
    print(
        f"Measuring local training time for {args.dataset} with {args.clients} clients..."
    )
    local_epochs = 5
    batch_size = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Load data and config for a single representative client (client_id=0)
    # We only need the trainloader and config for this task.
    trainloader, _, config = get_dataloaders(
        dataset_name=args.dataset,
        client_id=0,
        num_clients=args.clients,
        batch_size=batch_size,
    )
    num_samples = len(trainloader.dataset)

    # --- Timing Loop ---
    timings = []
    print(f"Running 3 trials of {local_epochs} local epochs each...")
    for i in range(3):
        # Re-initialize the model for a fair timing run each time
        model = get_model(dataset_name=args.dataset)

        start_time = time.perf_counter()
        
        # The train function now takes the config dictionary directly
        train(
            net=model,
            trainloader=trainloader,
            epochs=local_epochs,
            config=config,
        )
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        timings.append(elapsed)
        print(f"  Run {i+1}/3 finished in {elapsed:.4f} seconds.")

    # --- Report Results ---
    mean_time = np.mean(timings)
    std_time = np.std(timings)
    print("\n--- Results ---")
    print(f"Dataset: {args.dataset}")
    print(f"Samples / client: {num_samples}")
    print(
        f"Mean time for {local_epochs} epochs (3 runs): {mean_time:.4f}s (± {std_time:.4f}s)"
    )


if __name__ == "__main__":
    main()