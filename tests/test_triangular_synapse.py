import unittest
from typing import Collection, Tuple, Callable

import torch

from fuzzy.nn import TriangularSynapse
from fuzzy.member_f import TriangularMembF


DEBUG = True

if DEBUG:
    import matplotlib.pyplot as plt


class TestTriangularSynapseConstantInit(unittest.TestCase):
    def setUp(self) -> None:
        self.left = -3.0
        self.right = 3.0
        self.diameter = (self.right - self.left)
        self.count = 1
        self.module = TriangularSynapse(self.left, self.right, self.count, init_f=TriangularSynapse.all_hot_init)

    def test_forward_too_left(self):
        in_ = torch.Tensor([-6.0])
        out_ = self.module.forward(in_)
        expected = torch.Tensor([1.0])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))

    def test_forward_left_edge(self):
        in_ = torch.Tensor([self.left])
        out_ = self.module.forward(in_)
        expected = torch.Tensor([1.0])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))

    def test_forward_center(self):
        in_ = torch.Tensor([0.0])
        out_ = self.module.forward(in_)
        expected = torch.Tensor([1.0])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))

    def test_forward_right_edge(self):
        in_ = torch.Tensor([self.right])
        out_ = self.module.forward(in_)
        expected = torch.Tensor([1.0])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))

    def test_forward_too_right(self):
        in_ = torch.Tensor([+6.0])
        out_ = self.module.forward(in_)
        expected = torch.Tensor([1.0])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))

    def test_forward_rand(self):
        in_ = self.left + torch.rand(1) * self.diameter
        out_ = self.module.forward(in_)
        expected = torch.Tensor([1.0])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))

    def test_forward_rand_vector(self):
        in_ = self.left + torch.rand(5) * self.diameter
        out_ = self.module.forward(in_)
        expected = torch.Tensor([1.0, 1.0, 1.0, 1.0, 1.0])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))

    def test_forward_rand_matrix(self):
        in_ = self.left + torch.rand(5, 5) * self.diameter
        out_ = self.module.forward(in_)
        expected = torch.ones_like(in_)

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))

    def test_forward_rand_matrix_triangle_x3(self):
        layer = TriangularSynapse(self.left, self.right, count=3, init_f=TriangularSynapse.all_hot_init)

        in_ = self.left + torch.rand(10, 10) * self.diameter
        out_ = layer.forward(in_)
        expected = torch.ones_like(in_)

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))

    def test_forward_rand_matrix_triangle_x4(self):
        layer = TriangularSynapse(self.left, self.right, count=4, init_f=TriangularSynapse.all_hot_init)

        in_ = self.left + torch.rand(10, 10) * self.diameter
        out_ = layer.forward(in_)
        expected = torch.ones_like(in_)

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))


class TestTriangularSynapseRampInit(unittest.TestCase):
    def setUp(self) -> None:
        self.left = -3.0
        self.right = 3.0
        self.diameter = (self.right - self.left)
        self.count = 1
        self.module = TriangularSynapse(self.left, self.right, self.count, init_f=TriangularSynapse.ramp_init)

    def test_forward_rand_vector(self):
        step = 1 / 4
        eps = step / 100
        in_ = self.left + torch.arange(0.0, 1.0 + eps, step) * self.diameter
        out_ = self.module.forward(in_)
        expected = torch.Tensor([-1.0, -0.5, 0.0, +0.5, +1.0])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))


class TestTriangularSynapseInvRampInit(unittest.TestCase):
    def setUp(self) -> None:
        self.left = -3.0
        self.right = 3.0
        self.diameter = (self.right - self.left)
        self.count = 1
        self.module = TriangularSynapse(self.left, self.right, self.count, init_f=TriangularSynapse.inv_ramp_init)

    def test_forward_rand_vector(self):
        step = 1 / 4
        eps = step / 100
        in_ = self.left + torch.arange(0.0, 1.0 + eps, step) * self.diameter
        out_ = self.module.forward(in_)
        expected = torch.Tensor([+1.0, +0.5, 0.0, -0.5, -1.0])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))


class TestTriangularSynapseTrain(unittest.TestCase):
    def setUp(self) -> None:
        self.left = -3.0
        self.right = 3.0
        self.left_ds = self.left - 3.0
        self.right_ds = self.right + 3.0
        self.count = 1

    @classmethod
    def _learning_print_debug(cls, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor, layer: TriangularSynapse):
        print("x: {}, y: {}, y_hat: {}, gradient: {}, new values: {}".format(
            x, y, y_hat,
            layer._weights.grad, layer._weights.data
        ))

    @staticmethod
    def _gen_ds_item(
            base_function: Callable[[torch.Tensor], torch.Tensor],
            range_left: float, range_right: float, shape):
        assert range_left < range_right
        range_len = range_right - range_left
        x = range_left + torch.rand(*shape) * range_len
        y = base_function(x)
        return x, y

    @classmethod
    def _gen_dataset(cls, base_function, left, right, count, shape=(1,)):
        dataset = [
            cls._gen_ds_item(
                base_function, left, right, shape
            ) for _ in range(count)
        ]
        return dataset

    @classmethod
    def _train_layer(
            cls, layer: TriangularSynapse, dataset: Collection[Tuple[torch.Tensor, torch.Tensor]], lr: float = 1e-2,
            *, debug: bool = False
    ):
        error_fn = torch.nn.MSELoss()
        opt = torch.optim.RMSprop(
            params=layer.parameters(),
            lr=lr,
            alpha=0.9,  # default Keras
            momentum=0.0,  # default Keras
            eps=1e-7,  # default Keras
            centered=False  # default Keras
        )

        last_loss = None

        for x, y in dataset:
            y_hat = layer.forward(x)
            loss = error_fn(y_hat, target=y)

            opt.zero_grad()
            loss.backward()
            last_loss = loss.item()
            opt.step()

            if debug:
                cls._learning_print_debug(x, y, y_hat, layer)

        return last_loss

    def _assert_triangle_weights(self, layer):
        # Shall be approximately 0.0 (between -0.1 and +0.1)
        self.assertTrue(torch.greater_equal(
            layer._weights[0][0], torch.Tensor([-0.1]))
        )

        self.assertTrue(torch.less_equal(
            layer._weights[0][0], torch.Tensor([+0.1]))
        )

        # Shall be approximately 1.0 (between +0.9 and +1.1)
        self.assertTrue(torch.greater_equal(
            layer._weights[0][1], torch.Tensor([+0.9]))
        )

        self.assertTrue(torch.less_equal(
            layer._weights[0][1], torch.Tensor([+1.1]))
        )

        # Shall be approximately 0.0 (between -0.1 and +0.1)
        self.assertTrue(torch.greater_equal(
            layer._weights[0][2], torch.Tensor([-0.1]))
        )

        self.assertTrue(torch.less_equal(
            layer._weights[0][2], torch.Tensor([+0.1]))
        )

    def _gen_std_triangle(self) -> TriangularSynapse:
        return TriangularSynapse(self.left, self.right, self.count, init_f=TriangularSynapse.all_hot_init)

    def test_train_triangle_0d(self):
        function = TriangularMembF(3.0, 0.0)
        dataset = self._gen_dataset(function, self.left_ds, self.right_ds, 1000)
        layer = self._gen_std_triangle()

        self._train_layer(layer, dataset)
        self._assert_triangle_weights(layer)

    def test_train_triangle_1d(self):
        function = TriangularMembF(3.0, 0.0)
        dataset = self._gen_dataset(function, self.left_ds, self.right_ds, 250, shape=(4,))
        layer = self._gen_std_triangle()

        self._train_layer(layer, dataset, debug=DEBUG)
        self._assert_triangle_weights(layer)

    def test_train_triangle_2d(self):
        function = TriangularMembF(3.0, 0.0)
        dataset = self._gen_dataset(function, self.left_ds, self.right_ds, 250, shape=(2, 2))
        layer = self._gen_std_triangle()

        self._train_layer(layer, dataset)
        self._assert_triangle_weights(layer)

    @staticmethod
    def _visualize_layer(layer: TriangularSynapse, base_function, left, right, count: int):
        range_ = right - left
        step = range_ / count
        eps = step / 100

        x = torch.arange(start=left, end=right+eps, step=step)
        y = base_function(x)
        y_hat = layer.forward(x)

        x_view = x.numpy()
        y_view = y.numpy()
        y_hat_view = y_hat.numpy()

        plt.plot(x_view, y_view, 'r')
        plt.plot(x_view, y_hat_view, 'g')
        plt.show()

    @classmethod
    def _debug_view(cls, layer: TriangularSynapse, base_function, left, right, count: int):
        with torch.no_grad():
            print(layer._weights)
            cls._visualize_layer(layer, base_function, left, right, count)

    def test_train_cos_1d(self):
        left = - torch.pi * 3
        right = + torch.pi * 3
        count = 24

        function = torch.cos
        dataset = self._gen_dataset(function, left, right, 1, shape=(1000,))
        layer = TriangularSynapse(left, right, count, init_f=TriangularSynapse.random_init)

        last_loss = None

        for _ in range(100):
            last_loss = self._train_layer(layer, dataset)

        self.assertLessEqual(last_loss, 0.5)

        if DEBUG:
            self._debug_view(layer, function, left - torch.pi, right + torch.pi, 1000)


if __name__ == '__main__':
    unittest.main()
