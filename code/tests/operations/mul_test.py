from nnLib import Tensor
import numpy as np
import unittest

class MulTests(unittest.TestCase):
    def test_given_two_tensors_check_sum(self):
        ...
        x = Tensor(np.array([2]))
        y = Tensor(np.array([3]))

        z: Tensor = x * y

        self.assertAlmostEqual(z, Tensor.from_list([6.0])) 

        z.backward()
        self.assertIsNotNone(x.grad, "x is require_grad and backward is called, it should have gradient") 
        self.assertIsNotNone(y.grad, "x is require_grad and backward is called, it should have gradient")

        # dz/dx = 1*y
        self.assertEquals(x.grad, y)

        # dz/dy = x*1
        self.assertEquals(y.grad, x)