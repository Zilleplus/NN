from nnLib import Tensor
import numpy as np
import unittest

class AddTests(unittest.TestCase):
    def test_given_two_tensors_check_sum(self):
        x = Tensor(np.array([1]))
        y = Tensor(np.array([2]))

        z: Tensor = x + y

        self.assertEqual(z, Tensor.from_list([1.0 + 2.0])) 

        z.backward()
        self.assertIsNotNone(x.grad, "x is require_grad and backward is called, it should have gradient") 
        self.assertIsNotNone(y.grad, "y is require_grad and backward is called, it should have gradient")

        # dz/dx = 1
        self.assertEquals(x.grad, Tensor.from_list([1]))
        # dz/dy = 1
        self.assertEquals(y.grad, Tensor.from_list([1]))