def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Make divisible function for regression tasks.

    This function rounds the channel number down to the nearest value that can
    be divisible by the divisor, with adjustments suitable for regression tasks.

    Args:
        value (int): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int, optional): The minimum value of the output channel.
            Default: None, means that the minimum value equals to the divisor.
        min_ratio (float): The minimum ratio of the rounded channel
            number to the original channel number. Default: 0.9.

    Returns:
        int: The modified output channel number.
    """

    if min_value is None:
        min_value = divisor

    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)

    # Ensure that the new_value is not too far from the original value
    new_value = max(new_value, int(min_ratio * value))

    return new_value
