from typing import Sequence, Tuple

import torch.nn

from .triangular_synapse import TriangularSynapse


class LeNetFuzzy(torch.nn.Module):
    def __init__(
            self, *, flavor='MNIST', fuzzy_fcn: bool = True,
            mf_count: int = 12, mf_range: Tuple[float, float] = (-1.0, +1.0)
    ):
        super(LeNetFuzzy, self).__init__()

        if flavor == 'MNIST' or flavor == 'F-MNIST':
            self._init_as_ahaf_mnist()
        elif flavor == 'CIFAR10':
            self._init_as_ahaf_cifar()
        else:
            raise NotImplemented("Other flavors of LeNet-5 are not supported")

        # TODO: Check bias
        self.conv1 = torch.nn.Conv2d(
            in_channels=self._image_channels, out_channels=20, kernel_size=(5, 5),
            stride=(1, 1), padding=(0, 0), bias=False
        )

        self.act1 = torch.relu
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        # TODO: Check bias
        self.conv2 = torch.nn.Conv2d(
            in_channels=20, out_channels=50, kernel_size=(5, 5),
            stride=(1, 1), padding=(0, 0), bias=False
        )

        self.act2 = torch.relu
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self._flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)

        self.fc3 = torch.nn.Linear(
            in_features=self._fc3_in_features, out_features=self._fc4_in_features,
            bias=True
        )

        if fuzzy_fcn:
            self.act3 = TriangularSynapse(
                left=mf_range[0], right=mf_range[1], count=mf_count, input_dim=(self._fc4_in_features,)
            )
        else:
            self.act3 = torch.relu

        self.fc4 = torch.nn.Linear(
            in_features=self._fc4_in_features, out_features=10,
            bias=False
        )

        self._sequence = [
            self.conv1, self.act1, self.pool1,
            self.conv2, self.act2, self.pool2,
            self._flatten,
            self.fc3, self.act3,
            self.fc4
        ]

    def _init_as_ahaf_mnist(self):
        self._image_channels = 1
        self._fc3_in_features = 4 * 4 * 50
        self._fc4_in_features = 500

    def _init_as_ahaf_cifar(self):
        self._image_channels = 3
        self._fc3_in_features = 5 * 5 * 50
        self._fc4_in_features = 500

    def forward(self, x):
        for mod in self._sequence:
            x = mod(x)

        return x

    @property
    def act_params(self) -> Sequence[torch.nn.Parameter]:
        result = []

        if isinstance(self.act3, TriangularSynapse):
            result.extend(self.act3.parameters())

        return result

