# -*- coding: utf-8 -*-
# ========================================================
"""helper module is written for write useful function in indexer package"""
# ========================================================


# ========================================================
# Imports
# ========================================================
from pytorch_lightning.callbacks import ModelCheckpoint


def build_checkpoint_callback(save_top_k: int, dirpath: str, filename="QTag-{epoch:02d}-{val_loss:.2f}",
                              monitor="val_loss"):
    """

    :param save_top_k:
    :param filename:
    :param monitor:
    :return:
    """
    # saves a file like: input/QTag-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(monitor=monitor,  # monitored quantity
                                          filename=filename,
                                          save_top_k=save_top_k,  # save the top k models
                                          dirpath=dirpath,
                                          mode="min",  # mode of the monitored quantity for optimization
                                          )
    return checkpoint_callback
