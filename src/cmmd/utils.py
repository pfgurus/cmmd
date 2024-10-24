""" Various helpers. """


def range_255_1(x):
    """
    Convert from input range [0, 255] to [0, 1].
    """
    return x * (1 / 255)


def range_2_1(x):
    """
    Convert from input range [-1, 1] to [0, 1].
    """
    return x * 0.5 + 0.5