from nnLib import Tensor
import numpy as np
import unittest

class AddTests(unittest.TestCase):
    def test_given_two_tensors_check_sum(self):
        x = Tensor(np.array([1]))
        y = Tensor(np.array([2]))

        z: Tensor = x + y

        numpy_res = z.numpy()
        self.assertAlmostEqual(numpy_res[0], 1.0 + 2.0) 

        z.backward()
        self.assertIsNotNone(x.grad, "x is require_grad and backward is called, it should have gradient") 
        self.assertIsNotNone(y.grad, "x is require_grad and backward is called, it should have gradient")

        self.assertAlmostEquals(x.grad.numpy()[0], 1)
        self.assertAlmostEquals(y.grad.numpy()[0], 1)