from typing import Callable

import torch.nn


from ..member_f import TriangularMembF, LeftRampMembF, RightRampMembF


class TriangularLayer(torch.nn.Module):
    """
    Constant weights initialization, all membership functions active with the same weight of 1.0.
    """
    @staticmethod
    def all_hot_init(count: int) -> torch.Tensor:
        return torch.nn.Parameter(torch.ones(count + 2))

    def __init__(self, left: float, right: float, count: int, *, init_f: Callable[[int], torch.Tensor] = all_hot_init):
        super().__init__()
        self._mfs = []

        assert left < right
        assert count >= 1

        self._weights = init_f(count)
        self._mf_radius = (right - left) / count / 2

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
