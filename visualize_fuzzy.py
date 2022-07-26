import typing

import torch
import matplotlib.pyplot as plt

from fuzzy.libs import load_network, create_network
from fuzzy.nn import TriangularSynapse


def get_random_idxs(max_i, cnt=10) -> typing.Sequence:
    return [int(torch.randint(size=(1,), low=0, high=max_i)) for _ in range(cnt)]


def random_selection(params, idxs):
    return [params[i] for i in idxs]


def visualize_neuron(x: torch.Tensor, weights: torch.nn.Parameter, subfig):
    def _restore_weights(count: int, input_dim: typing.Tuple[int, ...]) -> torch.Tensor:
        return weights.data

    count_ = weights.size(dim=-1) - 2
    f = TriangularSynapse(left=-1.0, right=+1.0, count=count_, init_f=_restore_weights)
    y = f(x)

    x_view = x.cpu().numpy()
    y_view = y.cpu().numpy()

    subfig.plot(x_view, y_view)


def visualize_activations_fuzzy(fuzzy_params, fig, rows, show_subtitles=True):
    num_neurons = len(fuzzy_params) // 1
    start_index = max(0, num_neurons - rows)
    cols = 5

    x = torch.arange(start=-1.5, end=+1.5, step=0.1, device=fuzzy_params[0].device)

    gs = plt.GridSpec(rows, cols)

    for i in range(rows):
        param_idx = start_index + i
        all_mfs_weights = fuzzy_params[param_idx]
        sel = get_random_idxs(max_i=len(all_mfs_weights), cnt=cols)
        sel_mfs_weights = random_selection(all_mfs_weights, sel)

        for j in range(cols):
            subfig = fig.add_subplot(gs[i, j])
            if show_subtitles:
                subfig.set_title("L{} F{}".format(i, sel[j]))
            weights = sel_mfs_weights[j]
            visualize_neuron(x, weights, subfig=subfig)


def load_and_visualize(
        net_name: str, dataset_name: str, act_name: str, batch_size: int, start_epoch: int, frozen_act: bool,
        result_path, max_rows=None
):
    net = create_network(net_name, dataset_name)
    load_network(net, net_name, dataset_name, act_name, batch_size, start_epoch, frozen_act)

    fuzzy_params = net.act_params

    if net_name.endswith("Fuzzy"):
        params_per_neuron = 1
    else:
        raise NotImplemented("Other flavors of neural networks are not supported")

    num_neurons = len(fuzzy_params) // params_per_neuron
    show_subtitles = False

    if max_rows is None:
        rows = num_neurons
    else:
        rows = min(max_rows, num_neurons)

    height = 1.17 * (rows + 0.5)

    fig = plt.figure(tight_layout=True, figsize=(7, height))
    fig.suptitle("{}, {}, {}-like function".format(net_name, dataset_name, act_name))
    with torch.no_grad():
        if net_name.endswith("Fuzzy"):
            visualize_activations_fuzzy(fuzzy_params, fig, rows, show_subtitles)

    plt.savefig(result_path, dpi=300, format='svg')
    del net


def main():
    torch.manual_seed(seed=256)
    max_rows = 2

    fuzzy_init_as = "Ramp"

    load_and_visualize(
        net_name='LeNetFuzzy', dataset_name='F-MNIST', act_name=fuzzy_init_as, frozen_act=False,
        batch_size=64, start_epoch=100, result_path="runs/func_view_LeNet_F-MNIST_{}.svg".format(fuzzy_init_as),
        max_rows=max_rows
    )
    load_and_visualize(
        net_name='LeNetFuzzy', dataset_name='CIFAR10', act_name=fuzzy_init_as, frozen_act=False,
        batch_size=64, start_epoch=100, result_path="runs/func_view_LeNet_CIFAR10_{}.svg".format(fuzzy_init_as),
        max_rows=max_rows
    )
    load_and_visualize(
        net_name='KerasNetFuzzy', dataset_name='F-MNIST', act_name=fuzzy_init_as, frozen_act=False,
        batch_size=64, start_epoch=100, result_path="runs/func_view_KerasNet_F-MNIST_{}.svg".format(fuzzy_init_as),
        max_rows=max_rows
    )
    load_and_visualize(
        net_name='KerasNetFuzzy', dataset_name='CIFAR10', act_name=fuzzy_init_as, frozen_act=False,
        batch_size=64, start_epoch=100, result_path="runs/func_view_KerasNet_CIFAR10_{}.svg".format(fuzzy_init_as),
        max_rows=max_rows
    )


if __name__ == "__main__":
    main()
