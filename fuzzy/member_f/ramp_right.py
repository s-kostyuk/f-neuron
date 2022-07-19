import torch.nn


class RightRampMembF(torch.nn.Module):
    """
             ______
           /
          /
    _____/
    """
    def __init__(self, radius: float, center: float):
        super().__init__()

        self._radius = radius
        self._center = center
        self._left = center - radius

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only one dimension for now
        # TODO: Vectorize
        assert len(x.size()) == 1 and x.size(dim=0) == 1

        if x < self._left:
            return torch.Tensor([0.0])

        if self._left <= x <= self._center:
            return (x - self._left) / (self._center - self._left)

        if self._center < x:
            return torch.Tensor([1.0])

        assert False  # unreachable

    def __repr__(self):
        return "right: {},{}".format(self._left, self._center)
