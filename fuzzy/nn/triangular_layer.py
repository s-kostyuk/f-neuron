import torch.nn

from .triangular_synapse import TriangularSynapse


class TriangularLayer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, *, mf_count: int = 12):
        super().__init__()

        # Input image dimensions: [batch_size,n_channels,w,h]
        # Input image as vector:  [batch_size,n_channels*w*h]
        self._left = -1.0
        self._right = +1.0
        self._count = mf_count

        self._linear = torch.nn.Linear(in_features, out_features, bias)
        self._acts = torch.nn.ModuleList([
            TriangularSynapse(self._left, self._right, self._count) for _ in range(out_features)
        ])

    def _forward_synapse(self, x: torch.Tensor) -> torch.Tensor:
        x_vector = torch.unbind(x, dim=-1)

        assert len(x_vector) == self._linear.out_features
        assert len(x_vector) == len(self._acts)

        y_vector = [
            self._acts[i].forward(x_vector[i]) for i in range(self._linear.out_features)
        ]
        y = torch.stack(y_vector, dim=-1)

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear(x)
        x = self._forward_synapse(x)
        return x
