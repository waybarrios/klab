"""G7 confirmatory: end-to-end non-LLM PyTorch models, PrefetchSequential vs plain, both in CUDA graphs."""
import pathlib as _pathlib
KLAB = str(_pathlib.Path(__file__).resolve().parents[5]) if "__file__" in globals() else "."
import json, math, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, KLAB + "/research/l2-prefetch/src")
import harness as H, harness2, torch
from torch import nn
from api import PrefetchSequential
OUT = Path(KLAB + "/research/l2-prefetch/experiments/G7-non-llm-models/results/g7.json")


def cool(limit=60, wait=300):
    t0 = time.monotonic()
    while True:
        t = int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout)
        if t <= limit or time.monotonic() - t0 > wait: return t
        time.sleep(5)


def encoder(d, heads, ffn, n, norm_first=False):
    return [nn.TransformerEncoderLayer(d, heads, ffn, dropout=0.0, activation="gelu", batch_first=True, norm_first=norm_first)
            for _ in range(n)]


class PatchEmbed(nn.Module):
    def __init__(self, d=192):
        super().__init__()
        self.conv = nn.Conv2d(3, d, 16, 16); self.cls = nn.Parameter(torch.randn(1, 1, d)); self.pos = nn.Parameter(torch.randn(1, 197, d))
    def forward(self, img):
        x = self.conv(img).flatten(2).transpose(1, 2)
        return torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1) + self.pos


class Head(nn.Module):
    def __init__(self, d=192, classes=1000):
        super().__init__()
        self.norm = nn.LayerNorm(d); self.fc = nn.Linear(d, classes)
    def forward(self, x):
        return self.fc(self.norm(x)[:, 0])


def mlp(dims, act):
    return [nn.Sequential(nn.Linear(a, b), act()) for a, b in zip(dims[:-1], dims[1:])]


def models():
    bf = torch.bfloat16
    yield "minilm-l6-like (seq 32)", encoder(384, 12, 1536, 6), torch.randn(1, 32, 384), bf
    yield "bert-small-like (seq 16)", encoder(512, 8, 2048, 4), torch.randn(1, 16, 512), bf
    yield "vit-tiny-like (224px)", [PatchEmbed()] + encoder(192, 3, 768, 12, norm_first=True) + [Head()], torch.randn(1, 3, 224, 224), bf
    yield "dlrm-like mlp (batch 8)", mlp([13, 512, 256, 64], nn.ReLU) + mlp([64, 1024, 1024, 512, 256, 1], nn.ReLU), torch.randn(8, 13), torch.float32
    yield "rl actor-critic mlp (batch 1)", mlp([256, 1024, 1024, 1024, 64], nn.Tanh), torch.randn(1, 256), torch.float32


def main():
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(0)
    res = {"models": []}
    with H.lock():
        res["regime_before"] = H.require_p0()
        for name, layers, x, dt in models():
            layers = [m.cuda().to(dt).eval() for m in layers]
            x = x.cuda().to(dt)
            weight_bytes = sum(p.numel() * p.element_size() for m in layers for p in m.parameters())
            temp = cool()
            t0 = time.monotonic(); fast = PrefetchSequential(layers, x); tune_s = time.monotonic() - t0
            plain = nn.Sequential(*layers)
            temp = cool()
            rows = harness2.run_variants({"plain": plain, "plain_aa": plain, "prefetch": fast}, x, 1, baseline="plain")
            p = rows["prefetch"]
            out = {"model": name, "weight_bytes": weight_bytes, "plain_us": rows["plain"]["us_per_op"], "prefetch_us": p["us_per_op"],
                   "saved_pct": p["saved_pct"], "ci95": p["ci95"], "aa_pct": rows["plain_aa"]["saved_pct"], "bit_identical": p["bit_identical"],
                   "tune_seconds": tune_s, "temp_start": temp, "plans": fast.report}
            res["models"].append(out); OUT.write_text(json.dumps(res, indent=1, default=str))
            kept = sum(1 for g in fast.report if g["plan"])
            print(f"{name}: weights {weight_bytes / 2**20:.1f} MiB | plain {out['plain_us']:.1f}us prefetch {out['prefetch_us']:.1f}us "
                  f"saved {p['saved_pct']:+.2f}% ci {p['ci95'][0]:+.2f}..{p['ci95'][1]:+.2f} aa {out['aa_pct']:+.2f}% bits {p['bit_identical']} "
                  f"groups kept {kept}/{len(fast.report)} tune {tune_s:.1f}s temp {temp}C", flush=True)
            del layers, fast, plain; torch.cuda.empty_cache()
        res["regime_after"] = H.regime()
    ms = res["models"]
    geo = 100 * (1 - math.exp(sum(math.log(m["prefetch_us"] / m["plain_us"]) for m in ms) / len(ms)))
    crit = {"a": geo >= 5, "b": all(m["ci95"][1] >= -1 for m in ms), "c": all(m["bit_identical"] for m in ms)}
    res.update(geomean=geo, criteria=crit); OUT.write_text(json.dumps(res, indent=1, default=str))
    print(f"G7 geomean {geo:.2f}% criteria {crit} ->", "SUPPORTED" if all(crit.values()) else "NOT SUPPORTED")


if __name__ == "__main__":
    main()
