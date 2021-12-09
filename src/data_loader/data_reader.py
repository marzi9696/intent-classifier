# -*- coding: utf-8 -*-
# ========================================================
"""data_reader module is written for read files"""
# ========================================================


# ========================================================
# Imports
# ========================================================

import pandas as pd
import numpy as np


def read_csv(path: str) -> pd.DataFrame:
    """

    :param path:
    :return:
    """
    return pd.read_csv(path)


def read_npy(path: str) -> np.ndarray:
    """
    load a list of numpy elements into memory
    :param path:
    :return:
    """
    return np.load(path, allow_pickle=True)


def read_excel(path:str) -> pd.DataFrame:
    """
    :param path: where to load data from
    :return: pd.DataFrame
    """
    return pd.read_excel(path, engine="openpyxl")

