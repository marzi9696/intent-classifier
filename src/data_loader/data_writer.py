# -*- coding: utf-8 -*-
# ========================================================
"""data_writer module is written for write data in files"""
# ========================================================


# ========================================================
# Imports
# ========================================================
import numpy as np


def write_npy(path: str, data: list) -> None:
    """
    save a list of numpy elements into disk
    :param path:
    :param data:
    :return:
    """
    return np.save(path, data, allow_pickle=True)
