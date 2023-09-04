import argparse

import flwr
import numpy as np


def main(num_clients=1, num_rounds=1) -> None:
    strategy = flwr.server.strategy.FedAvgAndroid(
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
    )

    # Start Flower server
    hist = flwr.server.start_server(
        server_address="[::]:8080",
        config=flwr.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    return hist


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-c", "--clients", type=int, help="minimum number of clients", default=1
    )
    argParser.add_argument(
        "-r", "--rounds", type=int, help="number of rounds", default=1
    )
    args = argParser.parse_args()
    hist = main(args.clients, args.rounds)
