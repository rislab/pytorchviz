import torch
import time
from torchviz import make_dot, track_grad_fn, label_var, label_arg, label_ret

# @track_grad_fn("another_op")
def another_op(A, B):
    uid = str(time.time_ns())
    print(uid)
    A = label_arg(A, "A", uid)
    B = label_arg(B, "B", uid)
    return label_ret(A @ B, "A@B", uid)

def custom_operation(A, B, y):
    uid = str(time.time_ns())
    print(uid)
    A = label_arg(A, "A", uid)
    B = label_arg(B, "B", uid)
    y = label_arg(y, "y", uid)
    B = another_op(A, B)
    B = another_op(A, B)
    x = (A@B + y) @ B
    x = label_var(x, "x", uid)
    return label_ret(x, "x", uid), label_ret(y, "y", uid)


# Create tensors
A = torch.randn(3, 3, requires_grad=True)
B = torch.randn(3, 3, requires_grad=True)
y = torch.randn(3, 1, requires_grad=True)

# Apply custom operations
r = another_op(A, B)
x, z = custom_operation(A, B, r+y)
# x = another_op(A, B)
C = x + z
# C = label_var(C, "C")

# Compute loss
loss = C.sum()
loss.backward()

dot = make_dot(loss, params={"A": A, "B": B, "y": y})
dot.render('custom_computation_graph', format='png', view=True)