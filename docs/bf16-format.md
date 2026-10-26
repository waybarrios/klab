# Exact BF16 block storage, version 1

The CPU screen selected **delta**, using total bytes on Qwen layers 0–3 only. Tuning totals are 93,278,048 bytes for delta and 93,644,160 for the fifteen-exponent table candidate. The latter remains a storage-screen comparison, not a GPU implementation. Selection data: `results/bf16-storage-screen.json`; input identity: `.cache/real-weights/manifest.json` and its full upstream SHA-256.

This is a lossless storage encoding, not a new numeric precision. Preserve native BF16 bit patterns. Existing FP32 weights are not converted into this format. Related lossless BF16 formats already exist; see the joint design's prior-work section.

## Layout

- Input is a row-major N,K matrix of little-endian uint16 BF16 bits.
- Each row contains `ceil(K/64)` blocks. Pad a short last block by repeating its last valid weight; padding never contributes to the dot product or returned unpacked tensor.
- Payload is a byte array; metadata is one little-endian uint32 per block in row-major block order. Payload block starts are 32-byte aligned.
- Metadata bits 0–22 are the payload offset in 32-byte units; bits 23–30 contain the base exponent; bit 31 is the raw-block flag. The payload address range is at most 256 MiB per packed tensor. Reject larger payloads before upload; pack each ring copy separately.
- For a coded block, its minimum exponent is `base`. It is encodable when `max_exponent - base <= 15`. Store exactly 64 twelve-bit codes, 96 bytes, plus four bytes of metadata.
- For a raw block, store all 64 original uint16 values, 128 bytes, plus four bytes of metadata. Set bit 31 and use base zero. No bits are truncated, including signed zeros, subnormals, infinities or NaN payloads.
- No exponent table is stored in the selected delta representation. All metadata, tail padding and raw blocks count toward storage. Both 96-byte and 128-byte payloads naturally preserve 32-byte alignment.

## Twelve-bit code and decoding

For original uint16 bits `u`, let `exponent = (u >> 7) & 255` and `sm = (u & 127) | ((u >> 8) & 128)`. The twelve-bit code is `sm | ((exponent-base) << 8)`.

Concatenate codes in element order, least significant bits first. Code i starts at bit `12*i`. For each adjacent pair of codes a,b, write three bytes: `a & 255`, `((a >> 8) & 15) | ((b & 15) << 4)`, and `(b >> 4) & 255`.

Decode a code c to the original bits as `(c & 127) | ((c & 128) << 8) | ((base + (c >> 8)) << 7)`. Raw blocks read their original uint16 values directly. Reject malformed CPU metadata, out-of-range offsets, truncated payloads and decoded exponent overflow. GPU kernels accept only CPU-validated immutable buffers.

Widen to FP32 by placing the reconstructed sixteen bits in the upper half of a uint32 bit pattern. Raw and decoded projection templates must use the same multiply/accumulation/reduction order. FP32 output and one-final-round BF16 output are separate contracts. Bit-exact storage does not imply bit-identical output to a different cuBLAS reduction tree.

## Observed storage screen

Across the 50 retained real tensors, total bytes including metadata are 78.18–78.29% of original BF16 storage. Both large projection shapes and the output head pass the <=80% storage gate. This is approximately 21.7–21.8% fewer logical weight-storage bytes. It is not a measured latency gain. The GPU kernel remains gated on the measured streaming baseline/roofline screen.
