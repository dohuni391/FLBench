"""Flower ClientApp."""

import flwr as fl
from flwr.common import Context
from flwr.common.typing import NDArrays, Scalar

from .dataset import get_dataloaders
from .model import get_model, get_weights, set_weights, train

class FlowerClient(fl.client.NumPyClient):
    """A Flower client for federated learning."""
    def __init__(self, model, trainloader, img_key, label_key, local_epochs):
        self.model = model
        self.trainloader = trainloader
        self.img_key = img_key
        self.label_key = label_key
        self.local_epochs = local_epochs

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return get_weights(self.model)

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]):
        set_weights(self.model, parameters)
        train(self.model, self.trainloader, epochs=self.local_epochs, img_key=self.img_key, label_key=self.label_key)
        return get_weights(self.model), len(self.trainloader.dataset), {}

def client_fn(context: Context) -> fl.client.Client:
    """Create a Flower client instance for a given client ID."""
    # Get properties from context
    run_config = context.run_config
    partition_id = context.partition_id
    
    # Extract parameters from config
    dataset_name = str(run_config["dataset"])
    num_clients = int(run_config["num_clients"])
    local_epochs = int(run_config.get("local_epochs", 5))

    # Load model and data for this specific client
    model = get_model(dataset_name)
    trainloader, _, img_key, label_key = get_dataloaders(
        dataset_name, partition_id, num_clients
    )

    # Return a new FlowerClient instance
    return FlowerClient(model, trainloader, img_key, label_key, local_epochs).to_client()

# The ClientApp, which `flwr run` will instantiate using the `client_fn`.
app = fl.client.ClientApp(client_fn=client_fn)