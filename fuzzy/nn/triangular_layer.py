from typing import Tuple

import torch.nn

from .triangular_synapse import TriangularSynapse


class TriangularLayer(torch.nn.Module):
    def __init__(
            self, in_features: int, out_features: int, bias: bool = True,
            *, mf_count: int = 12, mf_range: Tuple[float, float] = (-1.0, +1.0),
            fuzzy_init: str = "Ramp"):
        super().__init__()

        # Input image dimensions: [batch_size,n_channels,w,h]
        # Input image as vector:  [batch_size,n_channels*w*h]
        self._left = mf_range[0]
        self._right = mf_range[1]
        self._count = mf_count

        self._linear = torch.nn.Linear(in_features, out_features, bias)
        self._acts = TriangularSynapse(
            self._left, self._right, self._count, input_dim=(out_features,),
            init_f=TriangularSynapse.get_init_f_by_name(fuzzy_init)
        )

    def _forward_synapse(self, x: torch.Tensor) -> torch.Tensor:
        x_vector = torch.unbind(x, dim=-1)

        assert len(x_vector) == self._linear.out_features
        y = self._acts(x)

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear(x)
        x = self._forward_synapse(x)
        return x
