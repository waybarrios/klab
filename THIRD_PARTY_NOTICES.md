# Third-party software and data

The MIT license covers the first-party code in this repository. External
libraries, source trees, firmware and model weights retain their own licenses.
They are not included in the public source snapshot.

The experiments use PyTorch, NVIDIA CUDA/NVRTC and cuBLAS/cuBLASLt. Their
libraries are installed separately. The checkpoint study uses original BF16
tensors from `Qwen/Qwen2.5-0.5B-Instruct`; its model revision and weight digest
are recorded in the experiment registration.

Optional comparisons refer to NWC, Turbo-Lossless and jart/matmul. The
gpu-unlock repository is a separate hardware-state experiment. Local copies,
kernel modules, firmware and prebuilt comparison binaries are excluded from
this repository. No driver or module installation is part of the normal klab
benchmark workflow.
