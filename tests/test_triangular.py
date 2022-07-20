import unittest

import torch

from fuzzy.member_f import TriangularMembF


class TestTriangularMemberFunction(unittest.TestCase):
    def setUp(self) -> None:
        self.radius = 1.0
        self.center = 0.0
        self.function = TriangularMembF(self.radius, self.center)

    def test_forward_too_left(self):
        in_ = torch.Tensor([self.center - self.radius * 2.0])
        out_ = self.function.forward(in_)
        expected = torch.Tensor([0.0])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))

    def test_forward_left_edge(self):
        in_ = torch.Tensor([self.center - self.radius])
        out_ = self.function.forward(in_)
        expected = torch.Tensor([0.0])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))

    def test_forward_center(self):
        in_ = torch.Tensor([self.center])
        out_ = self.function.forward(in_)
        expected = torch.Tensor([1.0])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))

    def test_forward_left(self):
        in_ = torch.Tensor([self.center - self.radius / 2.0])
        out_ = self.function.forward(in_)
        expected = torch.Tensor([0.5])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))

    def test_forward_left_rand(self):
        in_ = self.center - torch.rand(1) * self.radius
        out_ = self.function.forward(in_)
        expected = torch.Tensor([1.0 - torch.absolute(in_ - self.center) / self.radius])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))

    def test_forward_right_rand(self):
        in_ = self.center + torch.rand(1) * self.radius
        out_ = self.function.forward(in_)
        expected = torch.Tensor([1.0 - torch.absolute(in_ - self.center) / self.radius])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))

    def test_forward_right(self):
        in_ = torch.Tensor([self.center + self.radius / 2.0])
        out_ = self.function.forward(in_)
        expected = torch.Tensor([0.5])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))

    def test_forward_right_edge(self):
        in_ = torch.Tensor([self.center + self.radius])
        out_ = self.function.forward(in_)
        expected = torch.Tensor([0.0])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))

    def test_forward_too_right(self):
        in_ = torch.Tensor([self.center + self.radius * 2.0])
        out_ = self.function.forward(in_)
        expected = torch.Tensor([0.0])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))


if __name__ == '__main__':
    unittest.main()
