import torch
import inspect
from contextlib import ContextDecorator
from dataclasses import dataclass

grad_fn_metadata = {}

ENABLE_LABELS = True

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

class CustomOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, op_name):
        ctx.op_name = op_name  # Store name inside ctx
        return x.clone()  # Cloning ensures unique grad_fn

    @staticmethod
    def backward(ctx, grad_output):
        print(f"Backward pass of: {ctx.op_name}")
        return grad_output, None  # Preserve the chain

def label_arg(tensor, label, group=None):
    return label_val(tensor, label, group=group, caller_frame=2, is_arg=True, color="blanchedalmond")

def label_ret(tensor, label, group=None):
    return label_val(tensor, label, group=group, caller_frame=2, is_ret=True, color="salmon")

def label_val(tensor, label, group=None, caller_frame=1, is_arg=False, is_ret=False, color="orange"):
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
    print(f"Tracking {label}: grad_fn ID = {grad_fn_id}")
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

# custom decorator to track grad_fn with a custom name
def track_grad_fn(custom_name):
    def decorator(fn):

        # dynamically create a class with the custom_name as the class name
        # and static methods for forward and backward
        CustomGradFn = type(
            custom_name,
            (torch.autograd.Function,),
            {
                'forward': staticmethod(lambda ctx, *inputs: fn(*inputs)),
                'backward': staticmethod(lambda ctx, *grad_outputs: tuple(
                    grad_outputs[0] if input.requires_grad else None for input in ctx.saved_tensors
                )),
            }
        )

        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            if isinstance(result, torch.Tensor) and result.requires_grad:
                result = CustomGradFn.apply(*args)
            return result
        return wrapper
    return decorator

# Custom decorator to track grad_fn with a custom name
def better_track_grad_fn(custom_name):
    def decorator(fn):
        # Dynamically create a subclass of torch.autograd.Function
        class CustomGradFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *inputs):
                ctx.save_for_backward(*inputs)
                return fn(*inputs)

            @staticmethod
            def backward(ctx, *grad_outputs):
                inputs = ctx.saved_tensors
                grad_inputs = torch.autograd.grad(fn(*inputs), inputs, grad_outputs[0], retain_graph=True)
                return grad_inputs # return gradients for each input

            @staticmethod
            def backward(ctx, *grad_outputs):
                inputs = ctx.saved_tensors
                outputs = fn(*inputs)

                # Ensure outputs and grad_outputs are tuples
                if not isinstance(outputs, tuple):
                    outputs = (outputs,)
                if not isinstance(grad_outputs, tuple):
                    grad_outputs = (grad_outputs,)

                # Compute gradients
                grad_inputs = torch.autograd.grad(
                    outputs, inputs, grad_outputs, retain_graph=True, allow_unused=True
                )

                # Ensure backward returns a tuple
                return grad_inputs if len(grad_inputs) > 1 else (grad_inputs[0],)


        # Set the class name dynamically
        CustomGradFn.__name__ = custom_name

        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            if isinstance(result, torch.Tensor) and result.requires_grad:
                result = CustomGradFn.apply(*args)
                # store metadata
                grad_fn_metadata[id(result.grad_fn)] = custom_name
            return result

        return wrapper
    return decorator

###############################################################################

# Custom decorator to track grad_fn with a custom name
def track_grad_fn(custom_name):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            
            # Attach metadata if the result is a tensor with requires_grad
            if isinstance(result, torch.Tensor) and result.requires_grad:
                attach_grad_fn_metadata(result, custom_name)
            
            # If the result is a tuple, attach metadata to each tensor in the tuple
            elif isinstance(result, tuple):
                # print("Result is tuple")
                # print(result)
                result = tuple(
                    attach_grad_fn_metadata(tensor, custom_name) if isinstance(tensor, torch.Tensor) and tensor.requires_grad else tensor
                    for tensor in result
                )
            
            return result
        return wrapper
    return decorator

# Function to attach metadata to the grad_fn of a tensor
def attach_grad_fn_metadata(tensor, custom_name):
    if tensor.grad_fn:
        print(f"Tracking {custom_name} with key {id(tensor.grad_fn)}")
        grad_fn_metadata[id(tensor.grad_fn)] = custom_name
    return tensor

###############################################################################

class CustomGradName(ContextDecorator):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        # Store the current context name globally
        global current_custom_name
        self.prev_name = current_custom_name
        current_custom_name = self.name

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore previous name
        global current_custom_name
        current_custom_name = self.prev_name

# Initialize a global variable to store context name
current_custom_name = None

# Wrapper function to track grad_fn automatically
def track_grad_fn(tensor):
    if tensor.grad_fn is not None and current_custom_name:
        grad_fn_id = id(tensor.grad_fn)
        grad_fn_metadata[grad_fn_id] = current_custom_name
        print(f"Tracking {current_custom_name}: grad_fn ID = {grad_fn_id}")
    return tensor


