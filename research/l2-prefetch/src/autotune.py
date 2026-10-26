"""Capture-time autotune of the prefetch plan for a chain of identical layers (H2b protocol)."""
import harness as H
from l2prefetch import ChainRunner

SIZES = (None, 1 << 20, 512 << 10)
BLOCKS = (1, 2, 4, 8)


def candidates(nbytes):
    out, seen = {}, set()
    for b in BLOCKS:
        for s in SIZES:
            size = nbytes if s is None or s >= nbytes else s
            if (b, size) in seen:
                continue
            seen.add((b, size))
            out[f"b{b}_{size >> 10}k"] = {"blocks": b, "kernel": "l2_hint", "stride": 128, "nbytes": size}
    return out


def tune(ops, reads, x0, nbytes, rounds=9, min_lower_ci=1.0):
    """Returns (plan or None, report). ops/reads must use calibration-only weights."""
    cands = candidates(nbytes)
    variants = {"torch": ChainRunner(ops, reads)}
    for name, plan in cands.items():
        variants[name] = ChainRunner(ops, reads, [plan] * len(ops))
    rows = H.run_variants(variants, x0, len(ops), rounds=rounds)
    best = min(cands, key=lambda k: rows[k]["us_per_op"])
    keep = rows[best]["ci95"][0] > min_lower_ci and rows[best]["bit_identical"]
    report = {k: {"saved_pct": v["saved_pct"], "ci95": v["ci95"]} for k, v in rows.items()}
    return (cands[best] if keep else None), {"best": best, "kept": keep, "candidates": report}


def stage_bytes(reads):
    return sum(t.numel() * t.element_size() for t in reads)


def tune_positions(ops, reads, x0, period, rounds=7, passes=1, min_lower_ci=1.0, log=print):
    """Coordinate descent over the plan of each position in a repeating stage pattern (period stages)."""
    plans = [None] * period
    expand = lambda ps: [ps[i % period] for i in range(len(ops))]
    history = []
    for _ in range(passes):
        for pos in range(period):
            nbytes = stage_bytes(reads[pos])
            cands = candidates(nbytes)
            variants = {"current": ChainRunner(ops, reads, expand(plans))}
            if plans[pos] is not None:
                variants["off"] = ChainRunner(ops, reads, expand(plans[:pos] + [None] + plans[pos + 1:]))
            for name, plan in cands.items():
                variants[name] = ChainRunner(ops, reads, expand(plans[:pos] + [plan] + plans[pos + 1:]))
            rows = H.run_variants(variants, x0, len(ops), rounds=rounds, baseline="current")
            best = min((k for k in rows if k != "current"), key=lambda k: rows[k]["us_per_op"])
            if rows[best]["ci95"][0] > min_lower_ci and rows[best]["bit_identical"]:
                plans[pos] = None if best == "off" else cands[best]
            history.append({"pos": pos, "bytes": nbytes, "best": best, "saved_vs_current": rows[best]["saved_pct"],
                            "ci95": rows[best]["ci95"], "kept": plans[pos] is not None})
            log(f"  pos {pos} ({nbytes >> 10} KiB): best {best} {rows[best]['saved_pct']:+.2f}% -> plan {plans[pos]}")
    return plans, history
