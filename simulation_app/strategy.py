"""Flower strategy and evaluation function."""

import time
from typing import Dict, Optional, Tuple

import flwr as fl
from flwr.common import Metrics, NDArrays, Scalar, ndarrays_to_parameters

from .model import get_model, set_weights, test, DEVICE
from .dataset import get_centralized_testloader

def gen_evaluate_fn(dataset_name: str, testloader):
    """Generate the centralized evaluation function."""
    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        net = get_model(dataset_name)
        set_weights(net, parameters)
        
        # Determine img_key and label_key for the test function
        if dataset_name.lower() == "cifar10":
            img_key, label_key = "img", "label"
        else: # MNIST and FEMNIST
            img_key, label_key = "image", "character" if dataset_name.lower() == "femnist" else "label"

        loss, accuracy = test(net, testloader, img_key, label_key)
        print(f"Server-side evaluation, round {server_round}: loss {loss:.4f}, accuracy {accuracy:.4f}")
        return loss, {"accuracy": accuracy, "timestamp": time.time()}
    return evaluate

def get_strategy(dataset_name: str, num_clients: int, initial_parameters):
    """Return the FedAvg strategy for the simulation."""
    testloader = get_centralized_testloader(dataset_name)
    evaluate_fn = gen_evaluate_fn(dataset_name, testloader)
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        initial_parameters=ndarrays_to_parameters(initial_parameters),
    )
    return strategy