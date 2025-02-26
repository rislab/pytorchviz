import torch
import inspect
from dataclasses import dataclass
from typing import Dict
import itertools
import functools

@dataclass
class Metadata():
    label: str
    shape: tuple
    no_op: bool
    group: str
    caller_fn: str
    caller_file: str
    is_arg: bool = False
    is_ret: bool = False
    color: str = None

grad_fn_metadata: Dict[str, Metadata] = {}

ENABLE_LABELS = True

class CustomOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, op_name):
        ctx.op_name = op_name  # Store name inside ctx
        return x.clone()  # Cloning ensures unique grad_fn

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # Preserve the chain

def label_arg(tensor, label, group=None, caller_frame=2):
    return label_var(tensor, label, group=group, caller_frame=caller_frame, is_arg=True, color="blanchedalmond")

def label_ret(tensor, label, group=None, caller_frame=2):
    return label_var(tensor, label, group=group, caller_frame=caller_frame, is_ret=True, color="salmon")

def label_var(tensor, label, group=None, caller_frame=1, is_arg=False, is_ret=False, color="orange"):
    if not ENABLE_LABELS:
        return tensor

    # get name of caller function
    stack = inspect.stack()
    if len(stack) >= caller_frame:
        caller_fn = stack[caller_frame].function
        caller_file = stack[caller_frame].filename.split("/")[-1]
    else:
        caller_fn = stack[-1].function
        caller_file = stack[caller_frame].filename.split("/")[-1]
    if group is None:
        group = f"{caller_file}::{caller_fn}"

    no_op = False
    if tensor.grad_fn is None:
        # add a no_op grad_fn
        tensor = tensor * 1.
        no_op = True

    tensor = CustomOp.apply(tensor, label)
    grad_fn_id = id(tensor.grad_fn)
    # print(f"Tracking {label}: grad_fn ID = {grad_fn_id}")
    meta = Metadata(
        label=label, 
        shape=tensor.shape,
        no_op=no_op, 
        group=group,
        caller_fn=caller_fn,
        caller_file=caller_file,
        is_arg=is_arg,
        is_ret=is_ret,
        color=color
    )
    grad_fn_metadata[grad_fn_id] = meta
    return tensor

def add_call_count(func):
    """
    decorator to assign a unique counter to each function.
    note this will not work correctly for recursive function calls
    """
    if not hasattr(func, "_call_counter"):
        # attach a unique counter to the function
        func._call_counter = itertools.count(0)

    @functools.wraps(func) # this keeps our decorated fn name and attributes
    def wrapper(*args, **kwargs):
        # increment counter
        count = next(func._call_counter)
        # print(f"{func.__name__} - Call Count: {count}")
        return func(*args, **kwargs)

    return wrapper

def get_uid(func):
    if hasattr(func, "_call_counter"):
        file = inspect.getfile(func).split("/")[-1]
        uid = file + "::" + func.__name__ + " " + str(func._call_counter)
        return uid
    else:
        raise ValueError ("func needs to be decorated with @add_call_count")

def label_fn(*return_names):
    def decorator(func):
        func = add_call_count(func)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            assert all(isinstance(arg, torch.Tensor) for arg in args), type(args[0])#"All args must be tensors"
            assert all(isinstance(value, torch.Tensor) for value in kwargs.values()), "All args must be tensors"

            group_uid = get_uid(func)
            cf = 3

            # get the function's argument names
            signature = inspect.signature(func)
            param_names = list(signature.parameters.keys())

            # NOTE I did't use tuple() generator here because it adds to our call stack
            labelled_args = []
            for i, arg in enumerate(args):
                labelled_args.append(label_arg(arg, param_names[i], group=group_uid, caller_frame=cf))
            labelled_args = tuple(labelled_args)

            labelled_kwargs = []
            for key, value in kwargs.items():
                labelled_kwargs[key] = label_arg(value, key, group=group_uid, caller_frame=cf)

            result = func(*labelled_args, *labelled_kwargs)
            if isinstance(result, tuple):
                assert all(isinstance(r, torch.Tensor) for r in result), "All return values must be tensors"
                labels = return_names
                if len(return_names) == 0:
                    labels = [f"return_{i}" for i in range(len(result))]
                
                # NOTE I did't use tuple() generator here because it adds to our call stack
                labelled_result = []
                for i, r in enumerate(result):
                    labelled_result.append(label_ret(r, labels[i], group=group_uid, caller_frame=cf))
                labelled_result = tuple(labelled_result)

            elif isinstance(result, torch.Tensor):
                labels = return_names
                if len(return_names) == 0:
                    labels = ["return_0"]
                labelled_result = label_ret(result, labels[0], group=group_uid, caller_frame=cf)
            return labelled_result

        return wrapper
    return decorator
