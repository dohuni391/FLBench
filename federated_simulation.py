"""Federated Learning with customizable parameters and early stopping."""

import argparse
from typing import Dict, List, Tuple, Optional
import os

import torch
import numpy as np
import flwr as fl
from flwr.common import Metrics, NDArrays, Context, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from flwr.client import NumPyClient, Client
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation

import utils as model_utils

# Global variables to store configuration
GLOBAL_ARGS = None
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FlowerClient(NumPyClient):
    """Flower client implementing training and evaluation."""
    
    def __init__(self, dataset_name: str, partition_id: int, num_clients: int, batch_size: int):
        self.partition_id = partition_id
        self.model = model_utils.get_model(dataset_name)
        self.trainloader, self.valloader, _ = model_utils.get_dataloader(
            dataset_name, partition_id, num_clients, batch_size
        )
    
    def get_parameters(self, config: Dict) -> List[torch.Tensor]:
        print(f"[Client {self.partition_id}] get_parameters")
        return model_utils.get_parameters(self.model)
    
    def fit(self, parameters, config: Dict):
        server_round = config.get("server_round", 0)
        local_epochs = config.get("local_epochs", 1)
        
        print(f"[Client {self.partition_id}, round {server_round}] fit")
        # Convert NumPy ndarrays to PyTorch tensors
        parameters_tensors = [torch.from_numpy(ndarray) for ndarray in parameters]
        model_utils.set_parameters(self.model, parameters_tensors)
        model_utils.train(self.model, self.trainloader, local_epochs)
        return model_utils.get_parameters(self.model), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config: Dict):
        print(f"[Client {self.partition_id}] evaluate")
        # Convert NumPy ndarrays to PyTorch tensors
        parameters_tensors = [torch.from_numpy(ndarray) for ndarray in parameters]
        model_utils.set_parameters(self.model, parameters_tensors)
        loss, accuracy = model_utils.test(self.model, self.valloader)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}


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
    model = model_utils.get_model(GLOBAL_ARGS.dataset)
    testloader, _ = model_utils.get_centralized_testloader(GLOBAL_ARGS.dataset, GLOBAL_ARGS.batch_size)
    
    # Convert NumPy ndarrays to PyTorch tensors
    parameters_tensors = [torch.from_numpy(ndarray) for ndarray in parameters]
    model_utils.set_parameters(model, parameters_tensors)
    
    loss, accuracy = model_utils.test(model, testloader)
    print(f"Round {server_round} - Server-side evaluation: loss {loss:.4f}, accuracy {accuracy:.4f}")
    
    # Check if the desired accuracy is reached
    if accuracy >= GLOBAL_ARGS.desired_accuracy:
        print(f"🎉 Desired accuracy {GLOBAL_ARGS.desired_accuracy:.4f} reached! Stopping training.")
        # Save the final model
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), f"models/{GLOBAL_ARGS.dataset}_federated_model.pth")
        print(f"Model saved to models/{GLOBAL_ARGS.dataset}_federated_model.pth")
        return loss, {"accuracy": accuracy, "stop": True}
    
    return loss, {"accuracy": accuracy}


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
    
    # Create FedAvg strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=max(2, GLOBAL_ARGS.num_clients // 2),
        min_evaluate_clients=max(2, GLOBAL_ARGS.num_clients // 4),
        min_available_clients=GLOBAL_ARGS.num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=ndarrays_to_parameters(initial_parameters),
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=fit_config,
    )
        
    # Configure the server for the specified number of rounds
    config = ServerConfig(num_rounds=GLOBAL_ARGS.max_rounds)
    
    return ServerAppComponents(strategy=strategy, config=config)


def main():
    """Parse command-line arguments and start the simulation."""
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
        default=1,
        help="Number of local epochs on each client"
    )
    parser.add_argument(
        "--desired_accuracy", 
        type=float, 
        default=0.85,
        help="Desired accuracy to stop training"
    )
    parser.add_argument(
        "--max_rounds", 
        type=int, 
        default=50,
        help="Maximum number of rounds to train"
    )
    
    args = parser.parse_args()
    
    # Store args in global variable to access in server_fn
    global GLOBAL_ARGS
    GLOBAL_ARGS = args
    
    print(f"Starting federated learning with:")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Number of clients: {args.num_clients}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Local epochs: {args.local_epochs}")
    print(f"  - Desired accuracy: {args.desired_accuracy}")
    print(f"  - Maximum rounds: {args.max_rounds}")
    
    # Configure client resources
    use_gpu = torch.cuda.is_available()
    client_resources = {
        "num_cpus": 1,
        "num_gpus": 0.1 if use_gpu else 0.0,
    }
    
    # Create client and server apps
    client_app = fl.client.ClientApp(client_fn=client_fn)
    server_app = ServerApp(server_fn=server_fn)
    
    # Run the simulation
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=args.num_clients,
        backend_config={"client_resources": client_resources}
    )


if __name__ == "__main__":
    main()