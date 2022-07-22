import csv
import typing
import os

import torch
import torch.nn
import torch.optim
import torch.utils.data
import torchinfo
import torchvision.transforms
import torchvision.datasets

from fuzzy.nn import LeNetFuzzy
from fuzzy.libs import RunningStat

DynamicData = typing.List[typing.Dict[str, typing.Any]]


DEBUG = False
RESULTS_PATH = "runs"
SAVE_DYNAMICS_ENABLED = True
TRAIN_CLASSIC = False
TRAIN_FUZZY = True


def create_results_folder():
    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)


def save_dynamic_data(
        dynamic_data: DynamicData, net_name: str, dataset_name: str,
        act_name: str, batch_size: int, epochs: int, frozen: bool
):
    frozen_str = "" if frozen else "_ul"
    path = "{}/dynamics_{}{}_{}_{}_bs{}_ep{}.csv".format(
        RESULTS_PATH, net_name, frozen_str, dataset_name, act_name, batch_size, epochs
    )
    fields = "epoch", "train_loss_mean", "train_loss_var", "test_acc", "lr"

    with open(path, mode='w') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(dynamic_data)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using GPU computing unit")
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
        print("Cuda computing capability: {}.{}".format(*torch.cuda.get_device_capability(device)))
    else:
        print("Using CPU computing unit")
        device = torch.device('cpu')

    return device


def get_mnist_dataset(augment: bool = False) -> typing.Tuple[torch.utils.data.Dataset, ...]:
    if augment:
        augments = (
            # as in Keras - each second image is flipped
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            # assuming that the values from git.io/JuHV0 were used in arXiv 1801.09403
            torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        )
    else:
        augments = ()

    train_set = torchvision.datasets.FashionMNIST(
        root="./data/FashionMNIST",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            (torchvision.transforms.ToTensor(), *augments)
        )
    )

    test_set = torchvision.datasets.FashionMNIST(
        root="./data/FashionMNIST",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            (torchvision.transforms.ToTensor(),)
        )
    )

    return train_set, test_set


def get_cifar10_dataset(augment: bool = False) -> typing.Tuple[torch.utils.data.Dataset, ...]:
    if augment:
        augments = (
            # as in Keras - each second image is flipped
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            # assuming that the values from git.io/JuHV0 were used in arXiv 1801.09403
            torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        )
    else:
        augments = ()

    train_set = torchvision.datasets.CIFAR10(
        root="./data/cifar",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            (torchvision.transforms.ToTensor(), *augments)
        )
    )

    test_set = torchvision.datasets.CIFAR10(
        root="./data/cifar",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            (torchvision.transforms.ToTensor(),)
        )
    )

    return train_set, test_set


def train_non_dsu(batches, dev, net, error_fn, opt):
    loss_stat = RunningStat()

    for mb in batches:
        # Get output
        x, y = mb[0].to(dev), mb[1].to(dev)

        y_hat = net.forward(x)
        loss = error_fn(y_hat, target=y)
        loss_stat.push(loss.item())

        # Update parameters
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss_stat


def create_network(net_name: str, dataset_name: str):
    if net_name == "LeNetFuzzy":
        return LeNetFuzzy(flavor=dataset_name)
    else:
        raise NotImplemented("Networks other than LeNetFuzzy are not supported.")


def train_eval(
        net_name: str, dataset_name: str
) -> None:
    """
    Dataset A:
    - FashionMNIST
    - 50 000 images - train set
    - 10 000 images - eval set

    Dataset B:
    - CIFAR-10
    - 50 000 images - train set
    - 10 000 images - eval set

    Preprocessing (datasets A and B):
    - divide pixels by 255 (pre-done in the torchvision's dataset)
    - augment: random horizontal flip and image shifting

    Training:
    - RMSprop
    - lr = 10**-4
    - lr_decay_mb = 10**-6
    - batch_size = ???

    :return: None
    """
    batch_size = 64
    nb_start_ep = 0
    nb_epochs = 100
    rand_seed = 42

    if SAVE_DYNAMICS_ENABLED:
        create_results_folder()

    dynamic_data = []  # type: typing.List[typing.Dict]

    print("\nTraining {} network with {} activation on {} dataset with batch size {}".format(
        net_name, "ReLU", dataset_name, batch_size
    ))

    dev = get_device()
    torch.manual_seed(rand_seed)

    if dataset_name == 'F-MNIST':
        train_set, test_set = get_mnist_dataset(augment=True)
        input_size = (batch_size, 1, 28, 28)
    elif dataset_name == 'CIFAR10':
        train_set, test_set = get_cifar10_dataset(augment=True)
        input_size = (batch_size, 3, 32, 32)
    else:
        raise NotImplemented("Datasets other than Fashion-MNIST and CIFAR-10 are not supported")

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1000, num_workers=4
    )

    net = create_network(net_name, dataset_name)

    net.to(device=dev)

    torchinfo.summary(net, input_size=input_size)
    # print(net.dsu_param_sets)

    error_fn = torch.nn.CrossEntropyLoss()

    def build_opt(params):
        return torch.optim.RMSprop(
            params=params,
            lr=1e-4,
            alpha=0.9,  # default Keras
            momentum=0.0,  # default Keras
            eps=1e-7,  # default Keras
            centered=False  # default Keras
        )

    def inv_time_decay(step: int) -> float:
        """
        InverseTimeDecay in Keras, default decay formula used in keras.optimizers.Optimizer
        as per OptimizerV2._decayed_lr() - see https://git.io/JEKA6 and https://git.io/JEKx2.
        """
        decay_steps = 1  # update after each epoch
        decay_rate = 1.562e-3

        return 1.0 / (1.0 + decay_rate * step / decay_steps)

    def build_sched(optimizer):
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=inv_time_decay
        )

    opt = build_opt(net.parameters())
    sched = build_sched(opt)

    for epoch in range(nb_start_ep, nb_epochs):
        net.train()

        loss_stat = train_non_dsu(train_loader, dev, net, error_fn, opt)

        net.eval()

        with torch.no_grad():
            net.eval()
            test_total = 0
            test_correct = 0

            for batch in test_loader:
                x = batch[0].to(dev)
                y = batch[1].to(dev)
                y_hat = net(x)
                _, pred = torch.max(y_hat.data, 1)
                test_total += y.size(0)
                test_correct += (pred == y).sum().item()

            net.train()

            print("Train set loss stat: m={}, var={}".format(loss_stat.mean, loss_stat.variance))
            print("Epoch: {}. Test set accuracy: {:.2%}".format(epoch, test_correct / test_total))
            print("Current LR: {}".format(sched.get_last_lr()))

            if SAVE_DYNAMICS_ENABLED:
                dynamic_data.append({
                    "train_loss_mean": loss_stat.mean,
                    "train_loss_var": loss_stat.variance,
                    "test_acc": test_correct / test_total,
                    "lr": sched.get_last_lr(),
                    "epoch": epoch
                })

        # Classic - update LR on each epoch
        sched.step()

    if SAVE_DYNAMICS_ENABLED:
        save_dynamic_data(dynamic_data, net_name, dataset_name, "ReLU", batch_size, nb_epochs, False)


def main():
    if TRAIN_CLASSIC:
        train_eval(net_name='LeNet', dataset_name='F-MNIST')

    if TRAIN_FUZZY:
        train_eval(net_name='LeNetFuzzy', dataset_name='F-MNIST')


if __name__ == "__main__":
    main()
