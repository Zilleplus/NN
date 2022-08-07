from nnLib import Tensor
import numpy as np
import unittest

class MulTests(unittest.TestCase):
    def test_given_two_tensors_check_sum(self):
        ...
        x = Tensor(np.array([2]))
        y = Tensor(np.array([3]))

        z = x * y

        numpy_res = z.numpy()
        self.assertAlmostEqual(numpy_res[0], 6.0) 

        # z.backward()
        # self.assertIsNotNone(x.grad, "x is require_grad and backward is called, it should have gradient") 
        # self.assertIsNotNone(y.grad, "x is require_grad and backward is called, it should have gradient")
