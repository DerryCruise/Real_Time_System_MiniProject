"""
Mini Project 1.3 - Real-Time Jitter Analyzer Under Simulated System Noise
Real-Time Systems, Unit 1

Periodic task: T = 10 ms, job body ~2 ms of CPU, 250 jobs per experiment, drift-free release loop.
Deadline D = T, measured from the IDEAL release instant (a fixed, externally assigned absolute
deadline), so a late release eats into the slack and the hit fraction reflects it.

Three independently controllable disturbances (Section 1.6 stand-ins):
  1. background CPU-load thread, duty-cycle intensity   -> general interference / OS jitter
  2. random pre-job stall (probability, magnitude)      -> interrupt / DMA-stall-like disruption
  3. per-job execution-time variability (distribution)  -> cache / pipeline-like variation

Run:  python mp1_3_jitter_analyzer.py       (about 1.5 minutes; figures go to ./figures)
"""
import csv
import json
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rt_common import calibrate, busy_ms, BackgroundLoad, run_periodic, analyze

os.makedirs("figures", exist_ok=True)
MS = 1e3
T = 0.010                    # period: 10 ms
C_MS = 2.0                   # nominal job body: 2 ms
D = T                        # deadline = period, from the ideal release
N_JOBS = 250
SEED = 7


def experiment(label, mechanism, param, duty=0.0, p_delay=0.0, delay_ms=0.0, var_kind=None, var_param=0.0, seed=SEED):
    """Run one configuration and return a result row plus the raw jitter series."""
    rng = np.random.default_rng(seed)
    # per-job workload (ms): fixed, truncated normal, or heavy-tailed lognormal
    if var_kind == "normal":
        work = np.clip(rng.normal(C_MS, var_param * C_MS, N_JOBS), 0.1, 4.0 * C_MS)
    elif var_kind == "lognormal":
        work = np.clip(C_MS * np.exp(rng.normal(0.0, var_param, N_JOBS)), 0.1, 40.0)
    else:
        work = np.full(N_JOBS, C_MS)
    stall = (rng.random(N_JOBS) < p_delay) * (delay_ms / MS)           # seconds, 0 if no stall this job

    def pre_job(k):
        if stall[k] > 0:
            time.sleep(stall[k])                                         # interrupt/DMA-like delay before the job starts

    with BackgroundLoad(duty):
        rec = run_periodic(lambda k: busy_ms(work[k]), N_JOBS, T, pre_job=pre_job if p_delay > 0 else None)
    a = analyze(rec, D, "ideal")
    row = dict(label=label, mechanism=mechanism, param=param,
               rel_std_ms=a["rel_std"] * MS, rel_max_ms=a["rel_max"] * MS,
               comp_std_ms=a["comp_std"] * MS, comp_max_ms=a["comp_max"] * MS,
               hit_pct=a["hit_fraction"] * 100)
    return row, a["release_jitter"]


def sweep_plot(ax, rows, xlabel, title, xs=None, log_x=False):
    xs = xs if xs is not None else [r["param"] for r in rows]
    ax.plot(xs, [r["rel_std_ms"] for r in rows], "o-", color="tab:blue", lw=2, label="Release jitter std")
    ax.plot(xs, [r["rel_max_ms"] for r in rows], "s--", color="tab:cyan", lw=1.5, label="Release jitter max |.|")
    ax.plot(xs, [r["comp_std_ms"] for r in rows], "^:", color="tab:green", lw=1.5, label="Completion jitter std")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Jitter (ms)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    if log_x:
        ax.set_xscale("log")
    ax2 = ax.twinx()
    ax2.plot(xs, [r["hit_pct"] for r in rows], "d-", color="tab:red", lw=2, label="Deadline-hit fraction (%)")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Hit fraction (%)", color="tab:red")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)


if __name__ == "__main__":
    ipm = calibrate()
    print(f"Calibration: {ipm:,.0f} loop iterations/ms | T = {T*MS:.0f} ms, C = {C_MS} ms, D = {D*MS:.0f} ms (from ideal release), "
          f"{N_JOBS} jobs/experiment | GIL switch interval = {sys.getswitchinterval()*MS:.0f} ms")

    rows, series = [], {}

    def run(label, mech, param, **kw):
        row, rj = experiment(label, mech, param, **kw)
        rows.append(row)
        series[label] = rj
        print(f"  {label:34s} rel std {row['rel_std_ms']:7.3f}  rel max {row['rel_max_ms']:7.2f}  "
              f"comp std {row['comp_std_ms']:7.3f}  comp max {row['comp_max_ms']:7.2f}  hit {row['hit_pct']:6.1f} %")
        return row

    print("\nBASELINE (all disturbances off)")
    base = run("baseline", "baseline", 0)

    print("\nMECHANISM 1 - background CPU-load thread (duty cycle)")
    load_rows = [base] + [run(f"load duty={d:.2f}", "load", d, duty=d) for d in (0.25, 0.5, 0.75, 1.0)]

    print("\nMECHANISM 2a - random stall: probability sweep (magnitude 4 ms)")
    prob_rows = [base] + [run(f"stall p={p:.2f}, 4 ms", "stall_prob", p, p_delay=p, delay_ms=4.0) for p in (0.02, 0.05, 0.1, 0.2, 0.4)]
    print("\nMECHANISM 2b - random stall: magnitude sweep (probability 0.10)")
    mag_rows = [base] + [run(f"stall p=0.10, {m:g} ms", "stall_mag", m, p_delay=0.1, delay_ms=m) for m in (1, 2, 4, 8, 12)]

    print("\nMECHANISM 3a - execution-time variability: truncated normal (sigma as a fraction of C)")
    norm_rows = [base] + [run(f"normal sigma={s:.2f}C", "var_normal", s, var_kind="normal", var_param=s) for s in (0.1, 0.25, 0.5, 1.0)]
    print("\nMECHANISM 3b - execution-time variability: heavy-tailed lognormal (sigma of log)")
    logn_rows = [base] + [run(f"lognormal sigma={s:.2f}", "var_lognormal", s, var_kind="lognormal", var_param=s) for s in (0.25, 0.5, 1.0, 1.5)]

    # ------------------------------------------------------------------------
    # Required sweep plots, one figure per disturbance mechanism
    # ------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    sweep_plot(ax, load_rows, "Background-load duty cycle", "Mechanism 1: background CPU-load thread")
    plt.tight_layout(); plt.savefig("figures/rts3_fig1_sweep_load.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    sweep_plot(ax[0], prob_rows, "Stall probability per job (magnitude 4 ms)", "Mechanism 2a: stall probability sweep")
    sweep_plot(ax[1], mag_rows, "Stall magnitude (ms, probability 0.10)", "Mechanism 2b: stall magnitude sweep")
    plt.tight_layout(); plt.savefig("figures/rts3_fig2_sweep_stall.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    sweep_plot(ax[0], norm_rows, "sigma / C (truncated normal)", "Mechanism 3a: normal execution-time spread")
    sweep_plot(ax[1], logn_rows, "sigma of log (heavy-tailed lognormal)", "Mechanism 3b: heavy-tailed execution time")
    plt.tight_layout(); plt.savefig("figures/rts3_fig3_sweep_variability.png", dpi=150); plt.close()

    # ------------------------------------------------------------------------
    # Combined plot: jitter growth without a corresponding drop in hit fraction
    # ------------------------------------------------------------------------
    uniq = {r["label"]: r for r in rows}.values()
    base_std = base["rel_std_ms"]
    colors = {"baseline": "k", "load": "tab:red", "stall_prob": "tab:blue", "stall_mag": "tab:cyan",
              "var_normal": "tab:green", "var_lognormal": "tab:olive"}
    plt.figure(figsize=(8.5, 5.2))
    for r in uniq:
        plt.scatter(r["rel_std_ms"], r["hit_pct"], color=colors[r["mechanism"]], s=55, edgecolor="k", linewidth=0.5,
                    label=r["mechanism"])
    h, l = plt.gca().get_legend_handles_labels()
    seen = dict(zip(l, h))
    plt.legend(seen.values(), seen.keys(), fontsize=8, loc="lower left")
    plt.axhspan(99, 101, color="green", alpha=0.12)
    plt.axvline(5 * base_std, color="gray", ls="--")
    plt.text(5 * base_std * 1.05, 70, "5x baseline jitter", fontsize=8, color="gray")
    plt.xscale("log")
    plt.ylim(0, 105)
    plt.xlabel("Release jitter std (ms, log scale)")
    plt.ylabel("Deadline-hit fraction (%)")
    plt.title("All configurations: jitter can grow by orders of magnitude while the hit fraction stays near 100 %")
    plt.gca().title.set_fontsize(10)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig("figures/rts3_fig4_combined.png", dpi=150); plt.close()

    # Jitter traces for one setting of each mechanism
    fig, ax = plt.subplots(4, 1, figsize=(10, 7.5), sharex=True)
    pick = [("baseline", "Baseline"), ("load duty=0.50", "Load, duty 0.5"),
            ("stall p=0.10, 4 ms", "Stall p=0.10, 4 ms"), ("lognormal sigma=1.00", "Lognormal sigma=1.0 (execution time)")]
    ymax = max(series[k].max() for k, _ in pick) * MS * 1.1
    for a, (k, name) in zip(ax, pick):
        a.plot(series[k] * MS, lw=1, color="tab:blue")
        a.set_ylabel("Release jitter (ms)", fontsize=8)
        a.set_ylim(-0.2, ymax)
        a.set_title(name, fontsize=9, loc="left")
        a.grid(True, alpha=0.3)
    ax[-1].set_xlabel("Job index")
    plt.tight_layout(); plt.savefig("figures/rts3_fig5_jitter_traces.png", dpi=150); plt.close()

    # ------------------------------------------------------------------------
    # Results table and "high hit fraction, high jitter" configurations
    # ------------------------------------------------------------------------
    uniq = list(uniq)
    print("\nRESULTS TABLE")
    print(f"{'configuration':34s} {'rel std':>8s} {'rel max':>8s} {'comp std':>9s} {'comp max':>9s} {'hit %':>7s}")
    for r in uniq:
        print(f"{r['label']:34s} {r['rel_std_ms']:8.3f} {r['rel_max_ms']:8.2f} {r['comp_std_ms']:9.3f} {r['comp_max_ms']:9.2f} {r['hit_pct']:7.1f}")
    hi = [r for r in uniq if r["hit_pct"] >= 99.0 and r["rel_std_ms"] >= 5 * base_std and r["mechanism"] != "baseline"]
    print(f"\nConfigurations with hit fraction >= 99 % and release jitter std >= 5x baseline ({5*base_std:.3f} ms):")
    for r in hi:
        print(f"  {r['label']:34s} release jitter std {r['rel_std_ms']:.3f} ms ({r['rel_std_ms']/base_std:.0f}x baseline), hit {r['hit_pct']:.1f} %")
    if not hi:
        print("  (none in this run)")

    with open("results_mp1_3.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(uniq[0].keys()))
        w.writeheader()
        w.writerows(uniq)
    with open("results_mp1_3.json", "w") as f:
        json.dump(dict(rows=uniq, baseline_std_ms=base_std, high_hit_high_jitter=[r["label"] for r in hi]), f, indent=2)
    print("\nDone. Figures in ./figures, table in results_mp1_3.csv")
