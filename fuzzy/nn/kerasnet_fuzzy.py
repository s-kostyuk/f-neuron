from typing import Sequence, Tuple

import torch.nn

from .triangular_synapse import TriangularSynapse


class KerasNetFuzzy(torch.nn.Module):
    """
    KerasNet - CNN implementation evaluated in arXiv 1801.09403 **but** wih fuzzy activations.
    The model is based on the example CNN implementation from Keras 1.x: git.io/JuHV0.

    Architecture:

    - 2D convolution 32 x (3,3) with (1,1) padding
    - ReLU
    - 2D convolution 32 x (3,3) w/o padding
    - ReLU
    - max pooling (2,2)
    - dropout 25%
    - 2D convolution 64 x (3,3) with (1,1) padding
    - ReLU
    - 2D convolution 64 x (3,3) w/o padding
    - ReLU
    - max pooling (2,2)
    - dropout 25%
    - fully connected, out_features = 512
    - fuzzy activation
    - dropout 50%
    - fully connected, out_features = 10
    - softmax activation

    """
    def __init__(
            self, *, flavor='MNIST', fuzzy_fcn: bool = True,
            mf_count: int = 12, mf_range: Tuple[float, float] = (-1.0, +1.0)
    ):
        super(KerasNetFuzzy, self).__init__()

        if flavor == 'MNIST' or flavor == 'F-MNIST':
            self._init_as_fuzzy_mnist()
        elif flavor == 'CIFAR10':
            self._init_as_fuzzy_cifar()
        else:
            raise NotImplemented("Other flavors of KerasNet are not supported")

        self.conv1 = torch.nn.Conv2d(
            in_channels=self._image_channels, out_channels=32, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1), bias=True
        )
        self.act1 = torch.relu

        self.conv2 = torch.nn.Conv2d(
            in_channels=self.conv1.out_channels, out_channels=32, kernel_size=(3, 3),
            stride=(1, 1), padding=(0, 0), bias=True
        )
        self.act2 = torch.relu

        self.pool3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.drop3 = torch.nn.Dropout2d(p=0.25)

        self.conv4 = torch.nn.Conv2d(
            in_channels=self.conv2.out_channels, out_channels=64, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1), bias=True
        )
        self.act4 = torch.relu

        self.conv5 = torch.nn.Conv2d(
            in_channels=self.conv4.out_channels, out_channels=64, kernel_size=(3, 3),
            stride=(1, 1), padding=(0, 0), bias=True
        )
        self.act5 = torch.relu

        self.pool6 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.drop6 = torch.nn.Dropout2d(p=0.25)

        self._flatter = torch.nn.Flatten(start_dim=1, end_dim=-1)

        self.fc7 = torch.nn.Linear(
            in_features=self._fc7_in_features, out_features=self._fc8_out_features, bias=True
        )

        if fuzzy_fcn:
            self.act7 = TriangularSynapse(
                left=mf_range[0], right=mf_range[1], count=mf_count, input_dim=(self._fc8_out_features,)
            )
        else:
            self.act7 = torch.relu

        self.drop7 = torch.nn.Dropout2d(p=0.5)

        self.fc8 = torch.nn.Linear(
            in_features=self._fc8_out_features, out_features=10, bias=True
        )

        # softmax is embedded in pytorch's loss function

        self._sequence = [
            self.conv1, self.act1, self.conv2, self.act2, self.pool3, self.drop3,
            self.conv4, self.act4, self.conv5, self.act5, self.pool6, self.drop6,
            self._flatter,
            self.fc7, self.act7, self.drop7,
            self.fc8
        ]

    def _init_as_fuzzy_mnist(self):
        self._image_channels = 1
        self._fc7_in_features = 5 * 5 * 64
        self._fc8_out_features = 512
        self._act1_img_dims = (28, 28)
        self._act2_img_dims = (26, 26)
        self._act4_img_dims = (13, 13)
        self._act5_img_dims = (11, 11)

    def _init_as_fuzzy_cifar(self):
        self._image_channels = 3
        self._fc7_in_features = 6 * 6 * 64
        self._fc8_out_features = 512
        self._act1_img_dims = (32, 32)
        self._act2_img_dims = (30, 30)
        self._act4_img_dims = (15, 15)
        self._act5_img_dims = (13, 13)

    def forward(self, x):
        for mod in self._sequence:
            x = mod(x)

        return x

    @property
    def act_params(self) -> Sequence[torch.nn.Parameter]:
        result = []

        if isinstance(self.act7, TriangularSynapse):
            result.extend(self.act7.parameters())

        return result

