import os


def read_binary(file: str, num_read: int, size: int, endian: str = "little", signed: bool = False):
    """
    Reads integer values from binary files that are output of RSQSim

    :param file: file to read
    :param num_read: number of integers to read
    :param size: size of number to read (in bytes)
    :param endian: usually "little" unless we end up running on a non-standard system
    :param signed: include capacity for reading negative values (False if reading positive integers only)
    :return:
    """
    # Check that parameter supplied for endianness makes sense
    assert endian in ("little", "big"), "Must specify either 'big' or 'little' endian"
    assert os.path.exists(file)
    with open(file, "rb") as fid:
        # Container to store numbers as they are read
        number_list = []
        # Set counter to zero... indexing necessary because of setup
        count = 0
        # Read in required number of bytes
        byte = fid.read(size)
        #
        while count < num_read and byte != b"":
            byte_int = int.from_bytes(byte, byteorder=endian, signed=signed)
            number_list.append(byte_int)
            count += 1
            byte = fid.read(size)
    return number_list
