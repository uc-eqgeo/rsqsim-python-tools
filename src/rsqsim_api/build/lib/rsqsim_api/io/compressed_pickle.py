import bz2
import _pickle as c_pickle


def compressed_pickle(title, data):
    with bz2.BZ2File(title + ".pbz2", "w") as f:
        c_pickle.dump(data, f)


def decompress_pickle(file):
    data = bz2.BZ2File(file, "rb")
    data = c_pickle.load(data)
    return data
