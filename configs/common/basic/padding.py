import torch.nn as nn


def zero_pad_for_regression(*args, **kwargs):
    """
    Zero padding for regression tasks. This method adds padding with zeros around the image.

    Args:
        *args: Positional arguments for ZeroPad2d.
        **kwargs: Keyword arguments for ZeroPad2d.

    Returns:
        nn.ZeroPad2d: Zero padding layer.
    """
    return nn.ZeroPad2d(*args, **kwargs)


def reflect_pad_for_regression(*args, **kwargs):
    """
    Reflection padding for regression tasks. This method adds padding by reflecting the pixels near the edge.

    Args:
        *args: Positional arguments for ReflectionPad2d.
        **kwargs: Keyword arguments for ReflectionPad2d.

    Returns:
        nn.ReflectionPad2d: Reflection padding layer.
    """
    return nn.ReflectionPad2d(*args, **kwargs)


def replicate_pad_for_regression(*args, **kwargs):
    """
    Replication padding for regression tasks. This method adds padding by replicating the pixels near the edge.

    Args:
        *args: Positional arguments for ReplicationPad2d.
        **kwargs: Keyword arguments for ReplicationPad2d.

    Returns:
        nn.ReplicationPad2d: Replication padding layer.
    """
    return nn.ReplicationPad2d(*args, **kwargs)
