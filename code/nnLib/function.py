from typing import Any

class Function:
    tensors: list[Any]
    saved_tensors: list[Any]
    requires_grad: bool

    def __init__(self, tensors: list[Any], requires_grad: bool = True):
        self.tensors = tensors
        self.requires_grad = requires_grad
        self.saved_tensors = []

    def forward(self, x, y):
        raise NotImplementedError(f"The forward operation is not implemented in {type(self)}")

    def backward(self, output_gradient):
        raise NotImplementedError(f"The backward operation is not implemented in {type(self)}")

    def save_for_backward(self, *tensors) -> None:
        self.saved_tensors = list(tensors)

    @classmethod
    def apply(cls, *xs):
        requires_grad = any([x.requires_grad for x in xs])
        # Put the result into a tensor.
        ctx = cls(tensors = list(xs), requires_grad=requires_grad)
        out = ctx.forward(*xs)
        out.requires_grad = requires_grad

        # Put the auto diff context into the tensor -> used by autograd.
        out.ctx = ctx # type: ignore

        return out

class Add(Function):
     def forward(self, x, y):
         z = x.add_data(y)
         self.save_for_backward(x, y)
         return z
 
     def backward(self, output_gradient):
         # d(x+y)/dx = 1
         dx = output_gradient
         # d(x+y)/dy = 1
         dy = output_gradient
 
         return dx, dy

class Mul(Function):
     def forward(self, x, y):
        z = x.mul_data(y)
        self.save_for_backward(x, y)
        return z
 
     def backward(self, output_gradient):
        # d(x*y)/dx = 1*y
        dx = self.tensors[1].mul_data(output_gradient)
        # d(x*y)/dy = 1*x
        dy = self.tensors[0].mul_data(output_gradient) 
 
        return dx, dy