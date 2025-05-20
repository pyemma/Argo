"""
This module contains some common util functions used for
feature engineering and feature processing
"""


def padding(x, max_num):
    """
    padding 0 to the sequence and bump the original values with 1 to compensate for the 0 padding
    """
    if len(x) >= max_num:
        return [elem + 1 for elem in x[:max_num]]
    return x + [0] * (max_num - len(x))
