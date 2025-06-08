import functools

import torch as th
from torch import nn


def contextmanager_to_decorator(cm):
    def decorator(fn):
        @functools.wraps(fn)
        def newfn(*args, **kwargs):
            with cm():
                return fn(*args, **kwargs)

        return newfn

    return decorator


def default_device_type() -> str:
    if th.cuda.is_available():
        return "cuda"
    elif th.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


no_grad = contextmanager_to_decorator(th.no_grad)
DEFAULT_DEVICE = th.device(type=default_device_type())


def set_default_torch_device(device):
    global DEFAULT_DEVICE
    DEFAULT_DEVICE = th.device(device)


def dev():
    return DEFAULT_DEVICE


def zeros(*args, **kwargs):
    return th.zeros(*args, **kwargs, device=dev())


def ones(*args, **kwargs):
    return th.ones(*args, **kwargs, device=dev())


def arange(*args, **kwargs):
    return th.arange(*args, **kwargs, device=dev())


def NormedLinear(*args, scale=1.0, **kwargs):
    """
    nn.Linear but with normalized fan-in init
    """
    out = nn.Linear(*args, **kwargs)
    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    if kwargs.get("bias", True):
        out.bias.data *= 0
    return out


def LayerNorm(*args, **kwargs):
    out = nn.LayerNorm(*args, **kwargs)
    out.weight.no_scale = True
    return out


def flatten_image(x):
    """
    Flattens last three dims
    """
    *batch_shape, h, w, c = x.shape
    return x.reshape((*batch_shape, h * w * c))


def sequential(layers, x, *args, diag_name=None, use_checkpoint=False):
    for i, layer in enumerate(layers):
        x = layer(x, *args)
    return x


def save_kwargs(fn):
    """
    This decorator passes through the user-provided kwargs and adds one more, called
    save_kwargs, mapping to {"create_fn" : name_of_decorated_fn, "kwargs" : other_kwargs}

    You put on this decorator on a function that creates a pytorch module. This will
    save the kwargs and the function that was used to create the module.
    This lets us restore the model state later.
    """

    @functools.wraps(fn)
    def wrapper(**kwargs):
        if "save_kwargs" in kwargs:
            return fn(**kwargs)
        else:
            sk = {**kwargs, "create_fn": f"{fn.__module__}:{fn.__name__}"}
            return fn(save_kwargs=sk, **kwargs)

    return wrapper


def index(x, i):
    """
    Batched, broadcasting index of x along dimension i.ndim.

    For example, if x has shape (1, 2, 3, 4, 5) and i has shape (1, 1, 3)
    then the result has shape (1, 2, 3, 5) and each value in i must be between 0 and 3.
    """
    assert x.ndim >= i.ndim + 1
    gather_dim = i.ndim
    while i.ndim < x.ndim:
        i = i.unsqueeze(-1)
    expand_shape = list(x.shape)
    expand_shape[gather_dim] = 1
    i = i.expand(*expand_shape)
    xi = th.gather(x, gather_dim, i)
    assert xi.shape[gather_dim] == 1
    return xi.squeeze(gather_dim)
