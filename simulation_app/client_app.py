"""Flower ClientApp."""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from .utils import *

class FlowerClient(NumPyClient):
    """Flower client implementing federated learning for a PyTorch model."""

    def __init__(self, net, trainloader, valloader, config):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.config = config

    def fit(self, parameters, config):
        """Train the model on the local dataset."""
        set_weights(self.net, parameters)
        local_epochs = config["local_epochs"]
        train(self.net, self.trainloader, epochs=local_epochs, config=self.config)
        return get_weights(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the local validation set."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, config=self.config)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}


def client_fn(context: Context):
    """Create a Flower client instance for a given partition."""
    # Get properties from context
    partition_id = context.node_config["partition-id"]
    num_partitions = context.run_config["num_clients"]
    dataset_name = context.run_config["dataset_name"]
    local_epochs = context.run_config["local_epochs"]
    batch_size = context.run_config["batch_size"]

    # Load data partition
    trainloader, valloader, data_config = get_dataloaders(
        dataset_name, partition_id, num_partitions, batch_size
    )

    # Load model
    net = get_model(dataset_name, data_config["num_classes"])

    # Create a single Flower client instance
    return FlowerClient(net, trainloader, valloader, data_config).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)