from graphviz import Digraph
import torch
from torch.autograd import Variable

from .wrapper import grad_fn_metadata, Metadata

# Saved attrs for grad_fn (incl. saved variables) begin with `._saved_*`
SAVED_PREFIX = "_saved_"

def get_fn_name(fn, max_attr_chars=50):
    name = str(type(fn).__name__)
    if id(fn) not in grad_fn_metadata:
        return name

    # add a bunch of metadata
    meta = grad_fn_metadata[id(fn)]
    name = meta.label

    if meta.is_arg:
        attrs = {
            "Arg": tuple(meta.shape),
        }
    elif meta.is_ret:
        attrs = {
            "Ret": tuple(meta.shape),
        }
    else:
        attrs = {
            "shape": tuple(meta.shape),
        }

    max_attr_chars = max(max_attr_chars, 3)
    col1width = max(len(k) for k in attrs.keys())
    col2width = min(max(len(str(v)) for v in attrs.values()), max_attr_chars)
    sep = "-" * max(col1width + col2width + 2, len(name))
    attrstr = '%-' + str(col1width) + 's: %' + str(col2width)+ 's'
    truncate = lambda s: s[:col2width - 3] + "..." if len(s) > col2width else s
    params = '\n'.join(attrstr % (k, truncate(str(v))) for (k, v) in attrs.items())
    return name + '\n' + sep + '\n' + params


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph.

    If a node represents a backward function, it is gray. Otherwise, the node
    represents a tensor and is either blue or green:
     - Blue: reachable leaf tensors that requires grad (tensors whose `.grad`
         fields will be populated during `.backward()`)
     - Green: tensor passed in as outputs
     - Dark green: if any output is a view, we represent its base tensor with
         a dark green node.
    """
    global grad_fn_metadata
    print("IN MAKE DOT")
    print(grad_fn_metadata)

    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}
    else:
        param_map = {}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='10',
                     ranksep='0.1',
                     height='0.2',
                     fontname='monospace')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return "(" + (", ").join([str(v) for v in size]) + ")"

    def get_var_name(var, name=None):
        if not name:
            name = param_map[id(var)] if id(var) in param_map else ''
        return f"{name}\n{size_to_str(var.size())}"

    def add_nodes(fn, hide_no_ops=True):
        print(f"add_node {id(fn)} {fn}")
        assert not torch.is_tensor(fn)
        if fn in seen:
            return
        seen.add(fn)

        if hasattr(fn, 'variable'):
            # if grad_accumulator, add the node for `.variable`
            var = fn.variable
            seen.add(var)
            dot.node(str(id(var)), get_var_name(var), fillcolor='lightblue')
            dot.edge(str(id(var)), str(id(fn)))

        # add the node for this grad_fn
        # if not skip:
        if id(fn) in grad_fn_metadata:
            meta = grad_fn_metadata[id(fn)]
            dot.node(str(id(fn)), get_fn_name(fn), fillcolor=meta.color)
            if hide_no_ops and meta.no_op:
                # no_ops are created when we add a label_val() to a tensor that
                # has no grad_fn, so we introduce a no_op grad_fn so the label
                # appears in the computation graph

                # skip over the no_op and draw edges to the no_op's next_function
                next_fn = fn.next_functions[0][0]
                for v in next_fn.next_functions:
                    if v[0] is not None:
                        dot.edge(str(id(v[0])), str(id(fn)))
                        add_nodes(v[0])
                return

        else: 
            dot.node(str(id(fn)), get_fn_name(fn))

        # recurse
        if hasattr(fn, 'next_functions'):
            for u in fn.next_functions:
                if u[0] is not None:
                    dot.edge(str(id(u[0])), str(id(fn)))
                    add_nodes(u[0])

    def add_base_tensor(var, color='darkolivegreen1'):
        if var in seen:
            return
        seen.add(var)
        dot.node(str(id(var)), get_var_name(var), fillcolor=color)
        if (var.grad_fn):
            add_nodes(var.grad_fn)
            dot.edge(str(id(var.grad_fn)), str(id(var)))
        if var._is_view():
            add_base_tensor(var._base, color='darkolivegreen3')
            dot.edge(str(id(var._base)), str(id(var)), style="dotted")

    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_base_tensor(v)
    else:
        add_base_tensor(var)

    resize_graph(dot)

    # cleanup
    grad_fn_metadata = {}

    return dot

def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.

    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)
