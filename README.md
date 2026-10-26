# klab

klab is a small lab for GPU kernel experiments on a consumer laptop. It pairs CUDA C++ kernels (compiled at runtime
with NVRTC) with a C++ command-line launcher and Python tooling that times everything inside CUDA graphs against
strong vendor baselines. The current goal is to make small, latency-bound PyTorch work faster, in inference and in
training, without changing results.

This is a work in progress. Every number below comes from one RTX 3070 Ti Laptop GPU (GA104, SM 8.6, 46 SMs,
4 MiB of L2) with PyTorch 2.11 and CUDA 12.8. New experiments and corrections land here as they happen. A readable
walkthrough of the latest results is on the blog:
[Making small PyTorch ops faster on a laptop GPU](https://waybarrios.com/blog/small-ops-l2-prefetch/).

## What we have found so far

### 1. Check the power daemon before trusting any number

On this laptop, `nvidia-powerd` (the Dynamic Boost daemon) held the GPU in performance state P5 with memory at
810 MHz under any load, and `nvidia-smi --lock-memory-clocks` was accepted and ignored. Stopping the daemon gives
P0 at 6001 MHz: a memory copy goes from 35.8 GB/s to 356 GB/s and the same kernels run 2.3 to 7.4 times faster.
The GPU still idles at P8, and a reboot restores the daemon. No BIOS or firmware change is involved.

```sh
systemctl is-active nvidia-powerd
nvidia-smi --query-gpu=pstate,clocks.mem --format=csv
sudo systemctl stop nvidia-powerd     # undo with: sudo systemctl start nvidia-powerd
```

Older studies in this repository (the native-precision and lossless BF16 cohorts) were measured in the capped P5
regime. Their relative results stand as recorded, but their absolute times do not describe this GPU at full clock.

### 2. Results

| Result | Measured against | Outcome | Type |
|---|---|---|---|
| Cross-op L2 prefetch on held-out layer shapes | the same PyTorch chain without prefetch | 13.84% geomean latency saved; replicas 13.98% and 13.97%; 12.81% with a 2 s warm-up; outputs bit-identical | confirmatory |
| Small-batch training (forward + backward) | plain PyTorch autograd | 8.84% geomean on held-out shapes; gradients bit-identical | confirmatory |
| Full decode step, attention dispatch + prefetch | `torch.compile` elementwise + SDPA, one CUDA graph | 12.66% geomean on 12 held-out model x context configs, up to 30.6%; 12.36% with a cool-down before each config | confirmatory |
| Single-launch decode attention (v4), kernel only | copy + SDPA (flash split-KV) | 2.0 to 2.3 times faster at 512 tokens; slower at 32 heads x 1280+ tokens | exploratory |
| FlashInfer 0.7.2 single decode, kernel only | copy + SDPA and v4 | beats SDPA where supported; v4 beats it in 8 of 9 cells; no GQA group of 7 | exploratory |
| Three-way attention dispatch (SDPA / v4 / FlashInfer) | two-way dispatch | never slower on 12 held-out configs; FlashInfer picked in 3 | confirmatory |
| Small native GEMV kernels at P0 | best of cuBLAS, cuBLASLt and PyTorch | 17 to 38% faster for BF16 and FP16 shapes up to 512 square, 8% for BF16 at 1024 square, 4 to 22% for FP32 | exploratory, one process |
| MLP blocks (LayerNorm, Linear, GELU), batch 4 and 16 | plain PyTorch | 8 to 12% | exploratory |

Where it did not help: batch-1 convolutions (up to 3%), and end-to-end MiniLM-like, BERT-small-like, ViT-Tiny-like,
DLRM-like and RL policy models (between -4.5% and +1.3%). Their layers carry 3.4 to 6 MiB of weights, or 4 MiB for
FP32 1024 x 1024 layers, so the current and next layer never fit in 4 MiB of L2 together; the ViT also runs 197
tokens per layer. Two refuted predictions are worth knowing: a closed-form prefetch rule saved only 5.5% on held-out
shapes (the optimum near the L2 limit is too irregular to predict), and hand-written fused RMSNorm and SiLU kernels
lost to `torch.compile` by up to 5.3%.

### 3. How the two main pieces work

Cross-op L2 prefetch: while op `i` runs on the main stream, a small kernel on a side stream issues
`prefetch.global.L2` over the weights op `i+1` will read. The consumer kernels are unchanged and the prefetch never
writes memory, so outputs stay bit-identical. It pays when two consecutive layers fit in L2 (up to roughly 2.6 MiB
per layer on this GPU), when the batch is small, and when only a few prefetch blocks run. The plan for each layer
shape is chosen by timing candidates when the CUDA graph is captured, with plain PyTorch as the fallback.

Decode attention v4 handles one new token per sequence in a single launch: it appends the new key and value to the
cache, computes FP32 scores and softmax, splits the cache across blocks, and merges the partial results in the last
block to finish (an atomic ticket that resets itself, so the kernel can be replayed inside a CUDA graph). A variant
reads the cache length from device memory, so one captured graph serves a growing sequence.

## Setup

klab needs Linux, an NVIDIA GPU (SM 8.6 is the tested target), Python 3.12, a CUDA-enabled PyTorch build and a C++17
compiler. The kernels compile with NVRTC at runtime, so no CUDA toolkit install is needed for the main lab.
`requirements.lock` records the measured environment; the `+cu128` Torch build comes from PyTorch's CUDA wheel index.

```sh
git clone https://github.com/waybarrios/klab && cd klab
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu128
mkdir -p .scratch .cache
export TMPDIR="$PWD/.scratch" CUDA_CACHE_PATH="$PWD/.cache/cuda"
make                      # builds bin/klab and native/build/libkernel_lab.so
./bin/klab doctor         # GPU, runtime and compilation cache
./bin/klab check          # numerical checks, no performance claims
```

## Command-line interface

`./bin/klab help <command>` shows each command's options. Global options: `--wait` queues behind the GPU lock,
`--rebuild` recompiles the PTX, `--debug` prints more detail. Aliases: `bench` = `run`, `verify` = `check`,
`list` = `jobs`.

| Command | What it does |
|---|---|
| `doctor` | inspect the GPU, runtime and compilation cache |
| `problems`, `show <problem>` | list the local experiments; describe precision, baseline and constraints |
| `check [problem]` | numerical checks, no performance claims |
| `run [problem]` | run a benchmark (default: `fp32-projection`) |
| `compare` | compare baselines on common inputs |
| `tune` | search plain and lossless CUDA launch settings |
| `peak` | separate Tensor Core input-precision diagnostic |
| `stream` | validate frozen choices with rotating weights and A/A controls |
| `regime` | observe sustained bandwidth, launch floor and clocks |
| `corpus` | verify and retain selected real BF16 checkpoint tensors |
| `analyze-bf16` | screen exact BF16 storage formats on the CPU |
| `vendor-tune` | freeze a strong vendor pool and screen headroom |
| `probe-compressed` | tune-only probe of the real-weight lossless BF16 kernel |
| `jobs`, `status <run>`, `report <run>` | list runs, inspect one, write a Markdown comparison table |
| `factorial` | separate kernel gains from gpu-unlock gains |
| `build`, `version` | build the C++ launcher and CUDA driver wrapper; print the CLI version |
| `unlock-check` | read-only compatibility inventory for [gpu-unlock](https://github.com/Zagitta/gpu-unlock) |

Example: `./bin/klab run --shape 1,4096,4096 --seed 857 --rounds 31`.

## Python APIs

The L2 prefetch and decode attention code lives in `research/l2-prefetch/src`. Add that directory and the repository
root to `sys.path` to import it.

Prefetch for a list of modules, tuned on first construction:

```python
from api import PrefetchSequential

blocks = [nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 768), nn.GELU()) for _ in range(72)]
model = PrefetchSequential([b.cuda().bfloat16() for b in blocks], example_input)

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    y = model(static_x)       # same output bits as nn.Sequential(*blocks)
```

Prefetch hooks on an unmodified model, including the backward pass for training. Run eager warm-up steps on a side
stream before capturing backward; that is a PyTorch requirement for graph capture.

```python
from api import PrefetchHooks

def step(model, x):
    for p in model.parameters():
        p.grad = None
    model(x).float().square().mean().backward()

hooks = PrefetchHooks(model, x, backward=True, step=step)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    step(model, x)
hooks.remove()                # the model is plain PyTorch again
```

Decode attention for one new token (BF16, head dim 64, GQA), with the cache append fused in:

```python
from decode_attn import decode_attention, decode_attention_dynamic
from attn_dispatch import choose

out = decode_attention(q, k_new, v_new, k_cache, v_cache, length)            # k_cache: [kv_heads, max_len, 64]
out = decode_attention_dynamic(q, k_new, v_new, k_cache, v_cache, length_t)  # length_t: CUDA int32 tensor
impl, timings = choose(heads=9, kv_heads=3, length=512)                      # "custom" or "sdpa", measured
```

Older APIs at the repository root cover exact FP32, FP16 and BF16 projections (`kernel_lab.py`,
`half_projection.py`), lossless BF16 weight packing (`compressed_bf16.py`, format in `docs/bf16-format.md`), and
prepared projections with Torch custom operators (`prepared_bf16.py`, `native_prepared.py`). The scripts in
`examples/` show each of them on real tensors.

## Running the tests and experiments

```sh
python -m unittest discover -s tests -v                       # CPU suite
python -m unittest research/l2-prefetch/tests/test_prefetch.py \
                   research/l2-prefetch/tests/test_decode.py    # GPU suite, 8 tests
```

Each experiment in `research/l2-prefetch/experiments/<name>/` has its code in `code/` and its raw results in
`results/` (JSON and logs). Scripts hold the GPU lock in `.cache/gpu.lock` and refuse to time anything unless
`nvidia-powerd` is inactive and the laptop is on AC. For example:

```sh
python research/l2-prefetch/experiments/H2b-autotune/code/run.py          # held-out prefetch tuning
python research/l2-prefetch/experiments/H16-strong-baseline/code/run.py   # full decode step vs torch.compile + SDPA
```

The FlashInfer comparison needs FlashInfer and an nvcc that matches Torch's CUDA. Install them outside the measured
environment and point `KLAB_FI_SITE` (FlashInfer's install directory) and `KLAB_CUDA_HOME` (for example a conda env
with `cuda-nvcc=12.8`) at them; see `research/l2-prefetch/src/fi_env.py`. The JIT cache path must not contain spaces.

## How measurements are taken

Every timed operation is captured in a CUDA graph and replayed in randomized, paired rounds. Chains rotate through at
least 80 MiB of distinct weights, so each layer starts with a cold L2. Each comparison includes an A/A control (the
baseline timed twice), and savings are paired per round with bootstrap 95% confidence intervals. Short graphs get a
2 s time-based warm-up so the GPU is not measured during its clock ramp. Prefetch results are checked for bit
identity against plain PyTorch; new kernels are checked against an FP64 reference with tolerances tied to SDPA's own
error. For confirmatory tests the success criteria were fixed and hashed before the run, and refuted tests are kept.

## Repository layout

```
bin/, native/             C++ CLI and CUDA driver wrapper (built by make)
*.cu                      CUDA kernels compiled with NVRTC
kernel_lab.py, cuda_module.py, bench.py, experiment_control.py
                          kernel loading, CUDA graph timing, GPU lock
*_bf16.py, *_weights.py   exact and lossless BF16 / FP32 weight formats and projections
experiments/              registrations (JSON) for the native-precision and checkpoint studies
research/l2-prefetch/     cross-op L2 prefetch, decode attention, dispatch, APIs, tests, experiments and results
examples/, verification/  usage examples and GPU verification scripts
tests/                    CPU test suite
```

## Work in progress

Open items, roughly in order:

- GPUs with a larger L2 (Ada, Hopper), where layers that fail here may fit.
- Fewer bytes for larger models: the lossless BF16 kernels re-measured at P0, and models of 2B parameters and up.
- Head dimensions other than 64 in the decode attention kernel, and linear-attention layers.
- Wiring the prefetch and attention dispatch into the `klab` CLI.
- A strict thermal replication of the decode-step result (waiting for 60 C before each measurement) is running.

Issues are welcome, especially results from other GPUs and cases where something here makes things slower. Please
include the shape, the precision, the baseline and what you measured. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

First-party source is MIT licensed. External libraries and weights keep their own terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
