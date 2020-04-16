
def read_binary(file, fun, n, size, endian, signed):
    with open(file, "rb") as f:
        r = []
        count = 0
        byte = f.read(4)
        while count < n and byte != b"":
            i = int.from_bytes(byte, byteorder=endian, signed=signed)
            r.append(i)
            count += 1
            byte = f.read(4)
            byte = f.read(4)
    return r