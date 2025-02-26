import torch
import torchviz as tv

def test_no_labels():
    A = torch.randn(3, 3, requires_grad=True)
    B = torch.randn(3, 3, requires_grad=True)
    C = A @ B
    loss = C.sum()
    loss.backward()

    dot = tv.make_dot(loss, params={"A": A, "B": B})
    dot.render('test/test_no_labels', format='png', view=True, cleanup=True)

def test_label_var():
    A = torch.randn(3, 3, requires_grad=True)
    B = torch.randn(3, 3, requires_grad=True)
    C = A @ B
    # label a tensor
    C = tv.label_var(C, "C")
    loss = C.sum()

    dot = tv.make_dot(loss, params={"A": A, "B": B})
    dot.render('test/test_label_var', format='png', view=True, cleanup=True)

def test_label_args():
    A = torch.randn(3, 3, requires_grad=True)
    B = torch.randn(3, 3, requires_grad=True)
    # label the args
    A = tv.label_arg(A, "A")
    B = tv.label_arg(B, "B")
    C = A @ B
    C = tv.label_var(C, "C")
    loss = C.sum()

    dot = tv.make_dot(loss, params={"A": A, "B": B})
    dot.render('test/test_label_args', format='png', view=True, cleanup=True)

def test_label_args_rets():
    # labeling the args and return values of a function will automatically group
    # operations within that function together
    def foo(A, B):
        A = tv.label_arg(A, "A")
        B = tv.label_arg(B, "B")
        C = A @ B
        C = tv.label_ret(C, "C")
        return C

    A = torch.randn(3, 3, requires_grad=True)
    B = torch.randn(3, 3, requires_grad=True)
    C = foo(A, B)
    loss = C.sum()

    dot = tv.make_dot(loss, params={"A": A, "B": B})
    dot.render('test/test_label_args_rets', format='png', view=True, cleanup=True)

def test_nested():
    """
    label_arg and label_ret will also work with nested function calls 
    """

    # we need to add a call count so each label for foo() has a unique id
    # otherwise, all calls to foo will get grouped together in the graph
    @tv.add_call_count
    def bar(X, Y):
        uid = tv.get_uid(bar)
        X = tv.label_arg(X, "X", uid)
        Y = tv.label_arg(Y, "Y", uid)
        Z = X * Y
        Z = tv.label_ret(Z, "Y", uid)
        return Z

    def foo(A, B):
        A = tv.label_arg(A, "A")
        B = tv.label_arg(B, "B")
        C = bar(A,B) @ bar(A,B)
        C = tv.label_ret(C, "C")
        return C


    A = torch.randn(3, 3, requires_grad=True)
    B = torch.randn(3, 3, requires_grad=True)
    C = foo(A, B)
    loss = C.sum()

    dot = tv.make_dot(loss, params={"A": A, "B": B})
    dot.render('test/test_nested', format='png', view=True, cleanup=True)

def test_label_fn():
    """
    we can use tv.label_fn to automatically label_arg and label_ret as long as
    the args are tensors and the return values are also tensors
    """

    @tv.label_fn("A", "C")
    def foo(A, B):
        C = A @ B
        return C, A

    @tv.label_fn()
    def bar(X, Y):
        Z, X = foo(X, Y)
        X, _ = foo(X, Y)
        W = X * Z
        return W

    A = torch.randn(3, 3, requires_grad=True)
    B = torch.randn(3, 3, requires_grad=True)
    C = bar(A, B)
    loss = C.sum()

    dot = tv.make_dot(loss, params={"A": A, "B": B})
    dot.render('test/test_label_fn', format='png', view=True, cleanup=True)

if __name__ == "__main__":
    test_no_labels()
    test_label_var()
    test_label_args()
    test_label_args_rets()
    test_nested()
    test_label_fn()