import unittest

import torch

from fuzzy.member_f import RightRampMembF


class TestRightRampMemberFunction(unittest.TestCase):
    def setUp(self) -> None:
        self.radius = 1.0
        self.center = 0.0
        self.function = RightRampMembF(self.radius, self.center)

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
        expected = torch.Tensor([1.0])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))

    def test_forward_right(self):
        in_ = torch.Tensor([self.center + self.radius / 2.0])
        out_ = self.function.forward(in_)
        expected = torch.Tensor([1.0])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))

    def test_forward_right_edge(self):
        in_ = torch.Tensor([self.center + self.radius])
        out_ = self.function.forward(in_)
        expected = torch.Tensor([1.0])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))

    def test_forward_too_right(self):
        in_ = torch.Tensor([self.center + self.radius * 2.0])
        out_ = self.function.forward(in_)
        expected = torch.Tensor([1.0])

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertTrue(torch.allclose(out_, expected))

    @unittest.skip
    def test_forward_vector(self):
        in_ = torch.Tensor([
            -2.0, -1.5, -1.0, -0.5, 0.0, +0.5, +1.0, +1.5, +2.0
        ])
        expected = torch.Tensor([
            0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0
        ])
        out_ = self.function.forward(in_)

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertEqual(out_.size(), expected.size())
        self.assertTrue(torch.equal(out_, expected))

    @unittest.skip
    def test_forward_matrix(self):
        in_ = torch.Tensor([
            [-2.0, -1.5, -1.0],
            [-0.5, +0.0, +0.5],
            [+1.0, +1.5, +2.0]
        ])
        expected = torch.Tensor([
            [0.0, 0.0, 0.0],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ])
        out_ = self.function.forward(in_)

        self.assertTrue(isinstance(out_, torch.Tensor))
        self.assertEqual(out_.size(), expected.size())
        self.assertTrue(torch.equal(out_, expected))


if __name__ == '__main__':
    unittest.main()
