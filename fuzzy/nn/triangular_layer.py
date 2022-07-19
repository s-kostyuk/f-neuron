import torch.nn


from ..member_f import TriangularMembF, LeftRampMembF, RightRampMembF


class TriangularLayer(torch.nn.Module):
    def __init__(self, left: float, right: float, count: int, *, init: str = 'Constant'):
        super().__init__()
        self._mfs = []

        assert left < right
        assert count >= 1

        if init == 'Constant':
            self._weights = torch.nn.Parameter(
                torch.ones(count + 2, 1)
            )
        else:
            raise NotImplementedError()

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
        y = [mf.forward(x) for mf in self._mfs]
        y = torch.stack(y)
        y = torch.mul(self._weights, y)
        y = torch.sum(y)
        return y
