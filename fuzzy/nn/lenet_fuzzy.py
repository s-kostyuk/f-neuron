import torch.nn

from .triangular_layer import TriangularLayer


class LeNetFuzzy(torch.nn.Module):
    def __init__(self, *, flavor='MNIST'):
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

        self.fc3 = TriangularLayer(
            in_features=self._fc3_in_features, out_features=self._fc4_in_features,
            bias=True,
            mf_count=3  # TODO: Fix performance issues
        )

        self.fc4 = torch.nn.Linear(
            in_features=self._fc4_in_features, out_features=10,
            bias=False
        )

        self._sequence = [
            self.conv1, self.act1, self.pool1,
            self.conv2, self.act2, self.pool2,
            self._flatten,
            self.fc3,
            self.fc4
        ]

    def _init_as_ahaf_mnist(self):
        self._image_channels = 1
        self._fc3_in_features = 4 * 4 * 50
        # self._fc4_in_features = 500  # TODO: Fix performance issues
        self._fc4_in_features = 50

    def _init_as_ahaf_cifar(self):
        self._image_channels = 3
        self._fc3_in_features = 5 * 5 * 50
        self._fc4_in_features = 500

    def forward(self, x):
        for mod in self._sequence:
            x = mod(x)

        return x
