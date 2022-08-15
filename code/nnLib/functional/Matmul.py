from nnLib import Function

class MatMul(Function):
    def forward(self, l, r):
        z = l.matmul_data(r)

        self.save_for_backward(l, r)
        return z

    def backward(self, output_gradient):
        raise NotImplementedError()

        l = self.tensors[0]
        r = self.tensors[1]

        # TODO
        dl = None
        if l.requires_grad:
            dl = l.like_zeros()
        dr = None
        if r.requires_grad:
            dr = l.like_zeros()

        return dl, dr