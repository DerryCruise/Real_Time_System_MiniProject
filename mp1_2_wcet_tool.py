"""
Mini Project 1.2 - Measurement-Based WCET Estimation and Confidence Tool
Real-Time Systems, Unit 1

A small reusable tool:
  measure(func, inputs)         time every call individually with perf_counter(), keep all samples
  summarize(samples)            mean, std, min, max, p90/p95/p99 and an outlier flag
  recommended_wcet(max, margin) margined WCET-for-scheduling (60-70 % utilisation-budget convention)
  Campaign                      runs a RANDOM and an ADVERSARIAL input strategy for one target function

Three target functions of increasing structural complexity:
  1. linear_search    early-exit branch
  2. collatz_steps    data-dependent loop count
  3. packet_handler   nested conditionals inside a loop

Run:  python mp1_2_wcet_tool.py            (figures go to ./figures)
The tool reports an ESTIMATE together with its limitations. It never claims a proven bound.
"""
import gc
import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
perf = time.perf_counter
US = 1e6
SEED = 2026

# ============================================================================
# Part 1 - generic measurement harness (assumes nothing about the target's structure)
# ============================================================================


def measure(func, inputs, warmup=200):
    """Call func(inp) for every input; return an array of per-call durations in seconds."""
    for inp in inputs[:warmup]:                 # warm-up: caches, branch predictors, interpreter specialisation
        func(inp)
    out = np.empty(len(inputs))
    gc_was_on = gc.isenabled()
    gc.disable()                                # keep garbage-collector pauses out of the samples
    try:
        for i, inp in enumerate(inputs):
            t = perf()
            func(inp)
            out[i] = perf() - t
    finally:
        if gc_was_on:
            gc.enable()
    return out


# ============================================================================
# Part 2 - statistics and safety margin
# ============================================================================


def summarize(samples, outlier_ratio=2.0):
    """Required summary statistics plus a flag when max >> p99 (a rare outlier worth investigating)."""
    s = dict(n=len(samples), mean=float(np.mean(samples)), std=float(np.std(samples)), min=float(np.min(samples)),
             max=float(np.max(samples)), p90=float(np.percentile(samples, 90)), p95=float(np.percentile(samples, 95)),
             p99=float(np.percentile(samples, 99)))
    s["max_over_p99"] = s["max"] / s["p99"]
    s["outlier_flag"] = bool(s["max_over_p99"] > outlier_ratio)
    return s


def margin_from_utilization(u):
    """Margin implied by a utilisation budget u: WCET_sched = observed / u  (u = 0.65 -> +53.8 %)."""
    return 1.0 / u - 1.0


def recommended_wcet(observed_max, margin_pct):
    """Margined WCET-for-scheduling: observed maximum inflated by margin_pct percent."""
    return observed_max * (1.0 + margin_pct / 100.0)


# ============================================================================
# Part 3 - target functions
# ============================================================================
ARRAY = list(range(0, 4000, 2))                 # 2000 sorted even numbers


def linear_search(args):
    """Early-exit branch: returns as soon as the key is found."""
    arr, key = args
    for i, v in enumerate(arr):
        if v == key:
            return i
    return -1


def collatz_steps(n):
    """Data-dependent loop count: number of Collatz steps until n reaches 1."""
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def packet_handler(pkt):
    """Nested conditionals inside a loop; the deepest branch does extra work."""
    header_ok, kind, payload = pkt
    if not header_ok:                                   # path 1: drop immediately
        return 0
    acc = 0
    if kind == 0:                                       # path 2: heartbeat
        return len(payload)
    elif kind == 1:                                     # path 3: data, simple loop
        for v in payload:
            acc += v & 0xFF
        return acc % 251
    else:                                               # path 4: control, nested conditionals
        for v in payload:
            if v & 1:
                if v & 2:
                    if v & 4:                           # deepest branch: expensive
                        for _ in range(8):
                            acc = (acc * 31 + v) & 0xFFFF
                    else:
                        acc += v
                else:
                    acc -= v
            else:
                acc ^= v
        return acc


# ============================================================================
# Part 4 - input strategies (random/representative and hand-derived adversarial)
# ============================================================================


def longest_collatz_below(limit):
    """Offline analysis used to derive the adversarial input: exhaustive search with memoisation."""
    cache = {1: 0}
    best_n, best_steps = 1, 0
    for start in range(2, limit):
        n, path = start, []
        while n not in cache:
            path.append(n)
            n = n // 2 if n % 2 == 0 else 3 * n + 1
        base = cache[n]
        for i, p in enumerate(reversed(path), 1):
            cache[p] = base + i
        if cache[start] > best_steps:
            best_n, best_steps = start, cache[start]
    return best_n, best_steps


COLLATZ_LIMIT = 100_000
COLLATZ_WORST_N, COLLATZ_WORST_STEPS = longest_collatz_below(COLLATZ_LIMIT)


def random_inputs(target, rng, n):
    if target == "linear_search":                        # key present at a uniformly random position
        return [(ARRAY, ARRAY[rng.integers(0, len(ARRAY))]) for _ in range(n)]
    if target == "collatz_steps":
        return [int(v) for v in rng.integers(1, COLLATZ_LIMIT, n)]
    if target == "packet_handler":                        # mostly short/simple traffic with random bytes
        pkts = []
        for _ in range(n):
            ok = rng.random() < 0.95
            kind = int(rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15]))
            length = int(rng.integers(0, 256))
            pkts.append((ok, kind, [int(v) for v in rng.integers(0, 256, length)]))
        return pkts
    raise ValueError(target)


def adversarial_inputs(target, n):
    if target == "linear_search":                        # key absent: the loop scans the whole array
        return [(ARRAY, -1)] * n
    if target == "collatz_steps":                        # input with the longest chain in the domain
        return [COLLATZ_WORST_N] * n
    if target == "packet_handler":                        # valid header, control packet, every byte takes the deepest branch
        return [(True, 2, [0xFF] * 255)] * n
    raise ValueError(target)


TARGETS = {"linear_search": linear_search, "collatz_steps": collatz_steps, "packet_handler": packet_handler}
DESCRIPTION = {
    "linear_search": "early-exit branch",
    "collatz_steps": "data-dependent loop count",
    "packet_handler": "nested conditionals in a loop",
}


# ============================================================================
# Part 5 - campaign: run both strategies for one target and compare
# ============================================================================


class Campaign:
    def __init__(self, name, func, n_random=2000, n_adversarial=500, utilization=0.65, seed=SEED):
        self.name, self.func = name, func
        self.margin_pct = margin_from_utilization(utilization) * 100.0
        rng = np.random.default_rng(seed)
        self.rand_samples = measure(func, random_inputs(name, rng, n_random))
        self.adv_samples = measure(func, adversarial_inputs(name, n_adversarial))
        self.rand = summarize(self.rand_samples)
        self.adv = summarize(self.adv_samples)
        self.all_max = max(self.rand["max"], self.adv["max"])
        self.rec_random_only = recommended_wcet(self.rand["max"], self.margin_pct)
        self.rec_all = recommended_wcet(self.all_max, self.margin_pct)
        # robust comparison: adversarial *typical* (median) cost vs random maximum, not just one noisy max
        self.adv_median = float(np.median(self.adv_samples))
        self.adv_exceeds_random_max = self.adv["max"] > self.rand["max"]
        self.adv_median_exceeds_random_max = self.adv_median > self.rand["max"]
        self.underestimate_factor = self.adv_median / self.rand["max"]

    def row(self):
        return dict(target=self.name, structure=DESCRIPTION[self.name],
                    rand_mean_us=self.rand["mean"] * US, rand_p99_us=self.rand["p99"] * US, rand_max_us=self.rand["max"] * US,
                    adv_median_us=self.adv_median * US, adv_max_us=self.adv["max"] * US,
                    adv_max_exceeds_random_max=self.adv_exceeds_random_max,
                    adv_median_exceeds_random_max=self.adv_median_exceeds_random_max,
                    adv_median_over_random_max=self.underestimate_factor,
                    margin_pct=self.margin_pct, rec_random_only_us=self.rec_random_only * US, rec_all_us=self.rec_all * US,
                    rand_outlier_flag=self.rand["outlier_flag"], rand_max_over_p99=self.rand["max_over_p99"],
                    adv_outlier_flag=self.adv["outlier_flag"], adv_max_over_p99=self.adv["max_over_p99"])

    def plot(self, path):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        lo = min(self.rand["min"], self.adv["min"]) * US
        hi = max(self.rand["max"], self.adv["max"]) * US
        bins = np.logspace(np.log10(max(lo * 0.9, 1e-2)), np.log10(hi * 1.1), 70)
        ax.hist(self.rand_samples * US, bins=bins, color="tab:blue", alpha=0.75, label=f"Random strategy (n={self.rand['n']})")
        ax.hist(self.adv_samples * US, bins=bins, color="tab:red", alpha=0.75, label=f"Adversarial strategy (n={self.adv['n']})")
        ax.axvline(self.rand["max"] * US, color="navy", ls="--", label=f"random max {self.rand['max']*US:.1f} us")
        ax.axvline(self.all_max * US, color="darkred", ls="--", label=f"overall measured max {self.all_max*US:.1f} us")
        ax.axvline(self.rec_all * US, color="k", ls="-", lw=2, label=f"margined estimate {self.rec_all*US:.1f} us (+{self.margin_pct:.0f} %)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Execution time per call (us, log scale)")
        ax.set_ylabel("Calls (log scale)")
        ax.set_title(f"{self.name}: {DESCRIPTION[self.name]}")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()


if __name__ == "__main__":
    print(f"Adversarial input for collatz_steps below {COLLATZ_LIMIT}: n = {COLLATZ_WORST_N} ({COLLATZ_WORST_STEPS} steps), "
          f"found by exhaustive offline analysis")
    print(f"Safety margin from a 65 % utilisation budget: +{margin_from_utilization(0.65)*100:.1f} %  (WCET_sched = observed max / 0.65)")
    campaigns = [Campaign(n, f) for n, f in TARGETS.items()]
    rows = []
    for c in campaigns:
        r = c.row()
        rows.append(r)
        print(f"\n=== {c.name} ({DESCRIPTION[c.name]}) ===")
        print(f"  random     : mean {r['rand_mean_us']:8.2f} us  p90 {c.rand['p90']*US:8.2f}  p95 {c.rand['p95']*US:8.2f}  "
              f"p99 {r['rand_p99_us']:8.2f}  max {r['rand_max_us']:8.2f}  std {c.rand['std']*US:8.2f}  min {c.rand['min']*US:8.2f}")
        print(f"  adversarial: mean {c.adv['mean']*US:8.2f} us  p90 {c.adv['p90']*US:8.2f}  p95 {c.adv['p95']*US:8.2f}  "
              f"p99 {c.adv['p99']*US:8.2f}  max {r['adv_max_us']:8.2f}  median {r['adv_median_us']:8.2f}")
        print(f"  adversarial max exceeds random max : {r['adv_max_exceeds_random_max']}   "
              f"(adversarial median / random max = {r['adv_median_over_random_max']:.2f})")
        print(f"  margined estimate: from random only {r['rec_random_only_us']:.1f} us | from all samples {r['rec_all_us']:.1f} us")
        print(f"  outlier flag (max > 2 x p99): random {r['rand_outlier_flag']} ({r['rand_max_over_p99']:.1f}x), "
              f"adversarial {r['adv_outlier_flag']} ({r['adv_max_over_p99']:.1f}x)")
        c.plot(f"figures/rts2_hist_{c.name}.png")

    # Comparison table
    print("\nCOMPARISON TABLE (microseconds)")
    print(f"{'function':16s} {'random max':>11s} {'adv. median':>12s} {'adv. max':>10s} {'adv>rand max':>13s} {'margined (rand only)':>21s} {'margined (all)':>15s}")
    for r in rows:
        print(f"{r['target']:16s} {r['rand_max_us']:11.1f} {r['adv_median_us']:12.1f} {r['adv_max_us']:10.1f} "
              f"{str(r['adv_max_exceeds_random_max']):>13s} {r['rec_random_only_us']:21.1f} {r['rec_all_us']:15.1f}")

    # Running maximum of the random campaign versus the adversarial level: why more random samples may not help
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, c in zip(ax, campaigns):
        running = np.maximum.accumulate(c.rand_samples) * US
        a.plot(np.arange(1, len(running) + 1), running, color="tab:blue", lw=2, label="running max (random)")
        a.axhline(c.adv_median * US, color="tab:red", ls="--", label="adversarial median")
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("Number of random samples")
        a.set_ylabel("Observed maximum (us)")
        a.set_title(c.name, fontsize=10)
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("figures/rts2_running_max.png", dpi=150)
    plt.close()

    with open("results_mp1_2.json", "w") as f:
        json.dump(dict(collatz_worst_n=COLLATZ_WORST_N, collatz_worst_steps=COLLATZ_WORST_STEPS, rows=rows), f, indent=2)
    print("\nDone. Figures in ./figures, numbers in results_mp1_2.json")
