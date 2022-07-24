import typing

import torch


RESULTS_PATH = "runs"


def save_network(
        net: torch.nn.Module, net_name: str, dataset_name: str,
        act_name: str, batch_size: int, epochs: int, frozen: bool
):
    frozen_str = "" if frozen else "_ul"
    path = "{}/net_{}{}_{}_{}_bs{}_ep{}.bin".format(
        RESULTS_PATH, net_name, frozen_str, dataset_name, act_name, batch_size, epochs
    )
    torch.save(net.state_dict(), path)


def load_network(
        net: torch.nn.Module, net_name: str, dataset_name: str,
        act_name: str, batch_size: int, start_epoch: int, frozen: bool
):
    frozen_str = "" if frozen else "_ul"
    path = "{}/net_{}{}_{}_{}_bs{}_ep{}.bin".format(
        RESULTS_PATH, net_name, frozen_str, dataset_name, act_name, batch_size, start_epoch
    )
    net.load_state_dict(torch.load(path))
