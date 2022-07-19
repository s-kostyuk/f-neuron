import unittest
from typing import Collection, Tuple, Callable

import torch

from fuzzy.nn import TriangularLayer
from fuzzy.member_f import TriangularMembF


class TestTriangularLayerConstantInit(unittest.TestCase):
    def setUp(self) -> None:
        self.left = -3.0
        self.right = 3.0
        self.count = 1
        self.layer = TriangularLayer(self.left, self.right, self.count, init='Constant')

    def test_forward_too_left(self):
        in_ = torch.Tensor([-6.0])
        out_ = self.layer.forward(in_)
        expected = 1.0

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.eq(out_, expected))

    def test_forward_left_edge(self):
        in_ = torch.Tensor([self.left])
        out_ = self.layer.forward(in_)
        expected = 1.0

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.eq(out_, expected))

    def test_forward_center(self):
        in_ = torch.Tensor([0.0])
        out_ = self.layer.forward(in_)
        expected = 1.0

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.eq(out_, expected))

    def test_forward_right_edge(self):
        in_ = torch.Tensor([self.right])
        out_ = self.layer.forward(in_)
        expected = 1.0

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.eq(out_, expected))

    def test_forward_too_right(self):
        in_ = torch.Tensor([+6.0])
        out_ = self.layer.forward(in_)
        expected = 1.0

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.eq(out_, expected))

    def test_forward_rand(self):
        in_ = self.left + torch.rand(1) / 2
        out_ = self.layer.forward(in_)
        expected = 1.0

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.eq(out_, expected))


class TestTriangularLayerTrain1D(unittest.TestCase):
    def setUp(self) -> None:
        self.left = -3.0
        self.right = 3.0
        self.count = 1
        self.layer = TriangularLayer(self.left, self.right, self.count, init='Constant')

    @staticmethod
    def _learning_print_debug(x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor, layer: TriangularLayer):
        print("x: {}, y: {}, y_hat: {}, gradient: {}, new values: {}".format(
            x, y, y_hat,
            layer._weights.grad, layer._weights.data
        ))

    @staticmethod
    def _gen_ds_item(base_function: Callable[[torch.Tensor], torch.Tensor], range_left: float, range_right: float):
        assert range_left < range_right
        range_len = range_right - range_left
        x = range_left + torch.rand(1) * range_len
        y = base_function(x)
        return x, y

    @staticmethod
    def _train_layer(layer: TriangularLayer, dataset: Collection[Tuple[torch.Tensor, torch.Tensor]]):
        error_fn = torch.nn.MSELoss()
        opt = torch.optim.RMSprop(
            params=layer.parameters(),
            lr=1e-2,
            alpha=0.9,  # default Keras
            momentum=0.0,  # default Keras
            eps=1e-7,  # default Keras
            centered=False  # default Keras
        )

        for x, y in dataset:
            y_hat = layer.forward(x)
            loss = error_fn(y_hat, target=y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            # self._learning_print_debug(x, y, y_hat, self.layer)

    def _gen_dataset(self, base_function, count):
        dataset = [
            self._gen_ds_item(
                base_function, self.left - 3.0, self.right + 3.0
            ) for _ in range(count)
        ]
        return dataset

    def test_train_one_triangle(self):
        function = TriangularMembF(3.0, 0.0)
        dataset = self._gen_dataset(function, 1000)
        self._train_layer(self.layer, dataset)

        # Shall be approximately 0.0 (between -0.1 and +0.1)
        self.assertTrue(torch.greater_equal(
            self.layer._weights[0], torch.Tensor([-0.1]))
        )

        self.assertTrue(torch.less_equal(
            self.layer._weights[0], torch.Tensor([+0.1]))
        )

        # Shall be approximately 1.0 (between +0.9 and +1.1)
        self.assertTrue(torch.greater_equal(
            self.layer._weights[1], torch.Tensor([+0.9]))
        )

        self.assertTrue(torch.less_equal(
            self.layer._weights[0], torch.Tensor([+1.1]))
        )

        # Shall be approximately 0.0 (between -0.1 and +0.1)
        self.assertTrue(torch.greater_equal(
            self.layer._weights[2], torch.Tensor([-0.1]))
        )

        self.assertTrue(torch.less_equal(
            self.layer._weights[2], torch.Tensor([+0.1]))
        )


if __name__ == '__main__':
    unittest.main()
