from typing import Callable, Tuple

import torch.nn


from ..member_f import TriangularMembF, LeftRampMembF, RightRampMembF


class TriangularSynapse(torch.nn.Module):
    @staticmethod
    def ramp_init(count: int, input_dim: Tuple[int, ...] = (1,)) -> torch.Tensor:
        """
        Initialize member function weights to create a ramp function from -1.0 to +1.0.

        :param count: number of member functions.
        :param input_dim: input data dimensions:
            - scalar ``(1,)`` - by default;
            - vector ``(x,)``;
            - matrix ``(x,y)``;
            - multi-channel image: ``(z,x,y)`` where ``z`` is the number of channels.
        :return: initialized tensor of size ``(z,x,y,count)`` where dimensions ``z`` and ``y`` are optional.
        """
        low = -1.0
        high = +1.0
        range_ = high - low
        step = range_ / (count + 1)
        sample = torch.range(low, high, step)
        result = torch.empty(*input_dim, len(sample))
        return result.copy_(sample)

    @staticmethod
    def inv_ramp_init(count: int, input_dim: Tuple[int, ...] = (1,)) -> torch.Tensor:
        """
        Initialize member function weights to create an inverse ramp function from +1.0 to -1.0.

        :param count: number of member functions.
        :param input_dim: input data dimensions:
            - scalar ``(1,)`` - by default;
            - vector ``(x,)``;
            - matrix ``(x,y)``;
            - multi-channel image: ``(z,x,y)`` where ``z`` is the number of channels.
        :return: initialized tensor of size ``(z,x,y,count)`` where dimensions ``z`` and ``y`` are optional.
        """
        return - TriangularSynapse.ramp_init(count, input_dim)

    @staticmethod
    def random_init(count: int, input_dim: Tuple[int, ...] = (1,)) -> torch.Tensor:
        """
        Random weights initialization, ranging from -1.0 to +1.0.

        :param count: number of member functions.
        :param input_dim: input data dimensions:
            - scalar ``(1,)`` - by default;
            - vector ``(x,)``;
            - matrix ``(x,y)``;
            - multi-channel image: ``(z,x,y)`` where ``z`` is the number of channels.
        :return: initialized tensor of size ``(z,x,y,count)`` where dimensions ``z`` and ``y`` are optional.
        """
        low = -1.0
        high = +1.0
        range_ = high - low
        return low + torch.rand(*input_dim, count + 2) * range_

    @staticmethod
    def all_hot_init(count: int, input_dim: Tuple[int, ...] = (1,)) -> torch.Tensor:
        """
        Constant weights initialization, all membership functions active with the same weight of 1.0.

        :param count: number of member functions.
        :param input_dim: input data dimensions:
            - scalar ``(1,)`` - by default;
            - vector ``(x,)``;
            - matrix ``(x,y)``;
            - multi-channel image: ``(z,x,y)`` where ``z`` is the number of channels.
        :return: initialized tensor of size ``(z,x,y,count)`` where dimensions ``z`` and ``y`` are optional.
        """
        return torch.ones(*input_dim, count + 2)

    def __init__(
            self, left: float, right: float, count: int,
            *, init_f: Callable[[int, Tuple[int, ...]], torch.Tensor] = None, input_dim=(1,)
    ):
        super().__init__()
        self._mfs = torch.nn.ModuleList()

        assert left < right
        assert count >= 1

        if init_f is None:
            init_f = self.ramp_init

        self._weights = torch.nn.Parameter(init_f(count, input_dim))
        self._mf_radius = (right - left) / (count + 1)

        self._mfs.append(
            LeftRampMembF(self._mf_radius, left)
        )

        for i in range(1, count+1):
            mf_center = left + self._mf_radius * i
            mf = TriangularMembF(self._mf_radius, mf_center)

            self._mfs.append(mf)

        self._mfs.append(
            RightRampMembF(self._mf_radius, right)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = [mf.forward(x) for mf in self._mfs]
        x = torch.stack(x, -1)
        x = torch.mul(x, self._weights)
        x = torch.sum(x, -1)
        return x
