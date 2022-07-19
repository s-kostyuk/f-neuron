import torch.nn


class LeftRampMembF(torch.nn.Module):
    """
    -----
         \\
          \\
           \\_______
    """
    def __init__(self, radius: float, center: float):
        super().__init__()

        self._radius = radius
        self._center = center
        self._right = center + radius

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only one dimension for now
        # TODO: Vectorize
        assert len(x.size()) == 1 and x.size(dim=0) == 1

        if x < self._center:
            return torch.Tensor([1.0])

        if self._center <= x <= self._right:
            return (self._right - x) / (self._right - self._center)

        if self._right < x:
            return torch.Tensor([0.0])

        assert False  # unreachable

    def __repr__(self):
        return "left: {},{}".format(self._center, self._right)
