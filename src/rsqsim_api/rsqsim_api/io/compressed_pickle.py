"""Utilities for saving and loading Python objects as bz2-compressed pickles."""
import bz2
import _pickle as c_pickle


def compressed_pickle(title: str, data):
    """
    Serialise and compress a Python object to a bz2 pickle file.

    Parameters
    ----------
    title :
        Output file path without extension. The suffix ``.pbz2`` is
        appended automatically.
    data :
        Any picklable Python object to serialise.
    """
    with bz2.BZ2File(title + ".pbz2", "w") as f:
        c_pickle.dump(data, f)


def decompress_pickle(file: str):
    """
    Load and decompress a Python object from a bz2 pickle file.

    Parameters
    ----------
    file :
        Path to the ``.pbz2`` file to read.

    Returns
    -------
    object
        The deserialised Python object stored in the file.
    """
    data = bz2.BZ2File(file, "rb")
    data = c_pickle.load(data)
    return data
