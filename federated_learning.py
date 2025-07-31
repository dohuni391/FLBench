"""Federated Learning with customizable parameters, early stopping, and proper Ray resource management."""

import argparse
import os
import datetime
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import flwr as fl
from flwr.common import Metrics, NDArrays, Context, ndarrays_to_parameters, Parameters, FitRes
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.client import NumPyClient, Client
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation

import utils as model_utils

# Global variables to store configuration
GLOBAL_ARGS = None
DEVICE = model_utils.DEVICE

# Store metrics for visualization
ACCURACY_HISTORY = []
START_TIME = None
COMPLETION_ROUND = None

class FlowerClient(NumPyClient):
    """Flower client implementing training and evaluation."""
    
    def __init__(self, dataset_name: str, partition_id: int, num_clients: int, batch_size: int):
        self.partition_id = partition_id
        self.model = model_utils.get_model(dataset_name)
        self.trainloader, self.valloader, _ = model_utils.get_dataloader(
            dataset_name, partition_id, num_clients, batch_size
        )
    
    def get_parameters(self, config: Dict) -> List[torch.Tensor]:
        return model_utils.get_parameters(self.model)
    
    def fit(self, parameters, config: Dict):
        server_round = config.get("server_round", 0)
        local_epochs = config.get("local_epochs", 1)
        
        # Convert NumPy ndarrays to PyTorch tensors
        parameters_tensors = [torch.from_numpy(ndarray) for ndarray in parameters]
        model_utils.set_parameters(self.model, parameters_tensors)
        model_utils.train(self.model, self.trainloader, local_epochs)
        
        params = model_utils.get_parameters(self.model)
        dataset_size = len(self.trainloader.dataset)
        
        return params, dataset_size, {}
    
    def evaluate(self, parameters, config: Dict):
        # Convert NumPy ndarrays to PyTorch tensors
        parameters_tensors = [torch.from_numpy(ndarray) for ndarray in parameters]
        model_utils.set_parameters(self.model, parameters_tensors)
        loss, accuracy = model_utils.test(self.model, self.valloader)
        
        dataset_size = len(self.valloader.dataset)
        
        return float(loss), dataset_size, {"accuracy": float(accuracy)}


class FedAvgWithEarlyStopping(FedAvg):
    """Custom FedAvg strategy to implement early stopping."""
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        
        # Call the parent aggregate_fit method
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None and COMPLETION_ROUND is not None:
            # If the completion round is set, we can stop fitting
            print(f"Early stopping triggered in round {server_round}. No more fitting.")
            return None, {} # Returning None for parameters signals server to stop

        return aggregated_parameters, aggregated_metrics

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics from multiple clients using weighted average."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {"accuracy": sum(accuracies) / sum(examples)} if examples else {"accuracy": 0.0}


def fit_config(server_round: int) -> Dict[str, float]:
    """Return training configuration dict for each round."""
    return {
        "server_round": server_round,
        "local_epochs": GLOBAL_ARGS.local_epochs,
    }


def evaluate_fn(
    server_round: int,
    parameters: NDArrays,
    config: Dict[str, float],
) -> Optional[Tuple[float, Dict[str, float]]]:
    """Centralized evaluation function."""
    global COMPLETION_ROUND
    
    model = model_utils.get_model(GLOBAL_ARGS.dataset)
    testloader, _ = model_utils.get_centralized_testloader(GLOBAL_ARGS.dataset, GLOBAL_ARGS.batch_size)
    
    # Convert NumPy ndarrays to PyTorch tensors
    parameters_tensors = [torch.from_numpy(ndarray) for ndarray in parameters]
    model_utils.set_parameters(model, parameters_tensors)
    
    loss, accuracy = model_utils.test(model, testloader)
    time_elapsed = (datetime.datetime.now() - START_TIME)
    print(f"Round {server_round} at {time_elapsed.total_seconds()} seconds - Server-side evaluation: loss {loss:.4f}, accuracy {accuracy:.4f}")
    
    # Store accuracy and timestamp for visualization
    ACCURACY_HISTORY.append((server_round, float(accuracy), time_elapsed.total_seconds()))
    
    # Check if the desired accuracy is reached
    if accuracy >= GLOBAL_ARGS.desired_accuracy and COMPLETION_ROUND is None:
        COMPLETION_ROUND = server_round
        
        # Format the result in a very visible way
        print("\n" + "="*80)
        print(f"🎉 RESULT: Desired accuracy {GLOBAL_ARGS.desired_accuracy:.4f} reached in {server_round} rounds!")
        print(f"DATASET: {GLOBAL_ARGS.dataset}, CLIENTS: {GLOBAL_ARGS.num_clients}, LOCAL_EPOCHS: {GLOBAL_ARGS.local_epochs}")
        print(f"Time elapsed: {time_elapsed}")
        print("="*80 + "\n")
        
        # Save the final model
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{GLOBAL_ARGS.dataset}_{GLOBAL_ARGS.num_clients}clients_model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Generate plots
        generate_plots()

        print("\n--- COMPLETE ---")

    return loss, {"accuracy": accuracy}

def format_time(seconds, pos):
    """Formats seconds into HH:MM:SS or MM:SS format for plot ticks."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"

def generate_plots():
    """Generate and save accuracy plots in a single image."""
    os.makedirs("plots", exist_ok=True)

    if not ACCURACY_HISTORY:
        return

    # Extract data for plotting
    rounds, accuracies, timestamps = zip(*ACCURACY_HISTORY)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # --- Plot 1: Accuracy vs. Round ---
    ax1.plot(rounds, accuracies, 'o-', linewidth=2, markersize=6, label='Accuracy per Round')
    ax1.axhline(y=GLOBAL_ARGS.desired_accuracy, color='r', linestyle='--',
                label=f'Target Accuracy ({GLOBAL_ARGS.desired_accuracy:.4f})')

    if COMPLETION_ROUND is not None:
        completion_accuracy = next((acc for r, acc, _ in ACCURACY_HISTORY if r == COMPLETION_ROUND), None)
        if completion_accuracy:
            ax1.plot(COMPLETION_ROUND, completion_accuracy, 'ro', markersize=10,
                     label=f'Target reached (Round {COMPLETION_ROUND})')

    ax1.set_title(f'{GLOBAL_ARGS.dataset.upper()} - {GLOBAL_ARGS.num_clients} Clients - Accuracy vs. Round', fontsize=14)
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Plot 2: Accuracy vs. Timestamp ---
    ax2.plot(timestamps, accuracies, 'o-', linewidth=2, markersize=6, label='Accuracy over Time')
    ax2.axhline(y=GLOBAL_ARGS.desired_accuracy, color='r', linestyle='--',
                label=f'Target Accuracy ({GLOBAL_ARGS.desired_accuracy:.4f})')

    if COMPLETION_ROUND is not None:
        completion_info = next(((acc, ts) for r, acc, ts in ACCURACY_HISTORY if r == COMPLETION_ROUND), None)
        if completion_info:
            completion_accuracy, completion_timestamp = completion_info
            ax2.plot(completion_timestamp, completion_accuracy, 'ro', markersize=10,
                     label=f'Target reached (Round {COMPLETION_ROUND})')

    ax2.set_title(f'{GLOBAL_ARGS.dataset.upper()} - {GLOBAL_ARGS.num_clients} Clients - Accuracy vs. Time', fontsize=14)
    ax2.set_xlabel('Time (MM:SS)', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    ax2.xaxis.set_major_formatter(FuncFormatter(format_time))

    # Adjust layout and save the combined plot
    plt.tight_layout(pad=3.0)
    plot_path = f"plots/{GLOBAL_ARGS.dataset}_{GLOBAL_ARGS.num_clients}clients_accuracy_plots.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Combined accuracy plot saved to {plot_path}")
    plt.close()

def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""
    # Get partition ID from the context
    partition_id = context.node_config["partition-id"]
    
    # Create and return a Flower client
    return FlowerClient(
        dataset_name=GLOBAL_ARGS.dataset,
        partition_id=partition_id,
        num_clients=GLOBAL_ARGS.num_clients,
        batch_size=GLOBAL_ARGS.batch_size
    ).to_client()


def server_fn(context: Context) -> ServerAppComponents:
    """Construct server components with FedAvg strategy."""
    # Initialize model parameters
    init_model = model_utils.get_model(GLOBAL_ARGS.dataset)
    initial_parameters = model_utils.get_parameters(init_model)
    # Convert PyTorch tensors to NumPy ndarrays
    initial_parameters = [tensor.cpu().numpy() for tensor in initial_parameters]
    
    # Create our custom FedAvg strategy
    strategy = FedAvgWithEarlyStopping(
        fraction_fit=1.0,  # Use all clients as specified in the benchmark
        fraction_evaluate=0.5,
        min_fit_clients=GLOBAL_ARGS.num_clients,  # All clients must participate
        min_evaluate_clients=max(2, GLOBAL_ARGS.num_clients // 2),
        min_available_clients=GLOBAL_ARGS.num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=ndarrays_to_parameters(initial_parameters),
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=fit_config,
    )
        
    # Configure the server for the specified number of rounds
    max_rounds = 500
    if GLOBAL_ARGS.dataset == "femnist":
        max_rounds = 1200
    
    if GLOBAL_ARGS.max_rounds > 0:
        max_rounds = GLOBAL_ARGS.max_rounds
    
    config = ServerConfig(num_rounds=max_rounds)
    
    return ServerAppComponents(strategy=strategy, config=config)


def main():
    """Parse command-line arguments and start the simulation."""
    global START_TIME, GLOBAL_ARGS
    
    parser = argparse.ArgumentParser(description="Federated Learning with Early Stopping")
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="cifar10",
        choices=["mnist", "cifar10", "femnist"],
        help="Dataset to use (mnist, cifar10, femnist)"
    )
    parser.add_argument(
        "--num_clients", 
        type=int, 
        default=10,
        help="Number of clients to simulate"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--local_epochs", 
        type=int, 
        default=2,  # Default to 5 local epochs as per benchmark spec
        help="Number of local epochs on each client"
    )
    parser.add_argument(
        "--desired_accuracy", 
        type=float, 
        default=0.90,  # Default to 90% as per benchmark spec
        help="Desired accuracy to stop training"
    )
    parser.add_argument(
        "--max_rounds", 
        type=int, 
        default=50,  # 0 means use the default based on dataset
        help="Maximum number of rounds to train (0 for dataset default)"
    )
    parser.add_argument(
        "--cpu_per_client", 
        type=float, 
        default=2.0,
        help="Number of CPUs to allocate per client"
    )
    parser.add_argument(
        "--gpu_per_client", 
        type=float, 
        default=2.0,
        help="Fraction of GPU to allocate per client"
    )
    
    args = parser.parse_args()
    
    # Store args in global variable to access in server_fn
    GLOBAL_ARGS = args
    
    # Set max rounds based on dataset if not specified
    max_rounds = "500 (MNIST/CIFAR10) or 1200 (FEMNIST)"
    if args.max_rounds > 0:
        max_rounds = str(args.max_rounds)
    
    # Track start time
    START_TIME = datetime.datetime.now()
    
    print("\n" + "="*80)
    print(f"FEDERATED LEARNING BENCHMARK - {datetime.datetime.now().isoformat()}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of clients: {args.num_clients}")
    print(f"Batch size: {args.batch_size}")
    print(f"Local epochs: {args.local_epochs}")
    print(f"Desired accuracy: {args.desired_accuracy:.4f}")
    print(f"Maximum rounds: {max_rounds}")
    
    # Check and print device usage
    device_in_use = "GPU" if torch.cuda.is_available() and args.gpu_per_client > 0 else "CPU"
    print(f"Device in use: {device_in_use}")
    if device_in_use == "GPU":
        print(f"  GPU per client: {args.gpu_per_client}")
    
    # Configure client resources
    client_resources = {
        "num_cpus": args.cpu_per_client,
        "num_gpus": args.gpu_per_client if torch.cuda.is_available() else 0.0,
    }
    
    # Create client and server apps
    client_app = fl.client.ClientApp(client_fn=client_fn)
    server_app = ServerApp(server_fn=server_fn)
    
    # Run the simulation
    try:
        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=args.num_clients,
            backend_config={"client_resources": client_resources}
        )
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        # Print final summary
        print("\n" + "="*80)
        print("FINAL RESULTS:")
        
        if COMPLETION_ROUND is not None:
            print(f"✅ {args.dataset.upper()} with {args.num_clients} clients: Reached {args.desired_accuracy:.4f} accuracy in {COMPLETION_ROUND} rounds")
        else:
            max_accuracy = max([acc for _, acc, _ in ACCURACY_HISTORY]) if ACCURACY_HISTORY else 0
            max_round = len(ACCURACY_HISTORY)
            print(f"❌ {args.dataset.upper()} with {args.num_clients} clients: Failed to reach {args.desired_accuracy:.4f} accuracy")
            print(f"   Best accuracy: {max_accuracy:.4f} after {max_round} rounds")
        
        print(f"Total time: {datetime.datetime.now() - START_TIME}")
        print("="*80)
        
        # Generate plots if we have data but didn't reach completion
        if ACCURACY_HISTORY and COMPLETION_ROUND is None:
            generate_plots()

if __name__ == "__main__":
    main()