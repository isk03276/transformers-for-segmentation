from typing import Any

import numpy as np


def np_where_for_multi_dim_array(array_condition) -> np.ndarray:
    """
    Get index of a value in multi-dimension array.
    The 'array' arg must be an array of two or more dimensions.
    """
    return np.array(list(zip(*np.where(array_condition))))
