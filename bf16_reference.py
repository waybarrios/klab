"""Independent numpy decoder for docs/bf16-format.md, written by the parent.

This uses unpackbits instead of the encoder/GPU's shift-based decoding. It
does not import or call the packer. All values remain integer bit patterns.
"""

import numpy as np


def decode_bf16(payload, metadata, n, k):
    if type(n) is not int or type(k) is not int or min(n, k) <= 0 or n * k >= 2**31:
        raise ValueError("Expected positive N,K within signed 32-bit indexing")
    data = np.frombuffer(payload, dtype=np.uint8)
    words = np.asarray(metadata)
    blocks_per_row = (k + 63) // 64
    count = n * blocks_per_row
    if words.dtype != np.uint32 or words.ndim != 1 or len(words) != count:
        raise ValueError("Metadata must contain one uint32 per block")
    if data.size > 2**28:
        raise ValueError("Payload exceeds the version-1 offset range")
    raw = (words >> 31) != 0
    bases = (words >> 23) & 255
    offsets = (words & 0x7fffff).astype(np.int64) * 32
    sizes = np.where(raw, 128, 96)
    ends = np.cumsum(sizes, dtype=np.int64)
    if not np.array_equal(offsets, ends - sizes) or ends[-1] != data.size or np.any(raw & (bases != 0)):
        raise ValueError("Invalid block offsets, raw metadata or payload length")
    output = np.empty((count, 64), dtype=np.uint16)
    powers = 1 << np.arange(12, dtype=np.uint32)
    for start in range(0, count, 1024):
        stop = min(count, start + 1024)
        indices = np.arange(start, stop)
        coded_indices = indices[~raw[start:stop]]
        if len(coded_indices):
            byte_indices = offsets[coded_indices, None] + np.arange(96)
            stream_bits = np.unpackbits(data[byte_indices], axis=1, bitorder="little").reshape(-1, 64, 12)
            codes = (stream_bits * powers).sum(axis=2, dtype=np.uint32)
            exponents = bases[coded_indices, None] + (codes >> 8)
            if np.any(exponents > 255):
                raise ValueError("Decoded exponent overflows BF16")
            output[coded_indices] = ((codes & 127) | ((codes & 128) << 8) | (exponents << 7)).astype(np.uint16)
        raw_indices = indices[raw[start:stop]]
        if len(raw_indices):
            raw_bytes = data[offsets[raw_indices, None] + np.arange(128)]
            output[raw_indices] = raw_bytes.copy().view("<u2").reshape(-1, 64)
    return output.reshape(n, blocks_per_row * 64)[:, :k]
